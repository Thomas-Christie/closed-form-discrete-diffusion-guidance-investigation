"""Feynman-Kac steering of a masked diffusion process.

Implements the sequential Monte-Carlo scheme of Singhal et al.: a population of
particles is denoised together, and at scheduled steps each is scored, weighted by a
potential, and resampled. Rewards are typically defined only on complete sequences, so a
partially masked particle is scored by drawing completions of it and reducing their
rewards to one number - see :func:`monte_carlo_reward`.

Reference:
    R. Singhal, Z. Horvitz, R. Teehan, M. Ren, Z. Yu, K. McKeown and R. Ranganath,
    "A General Framework for Inference-time Scaling and Steering of Diffusion Models",
    ICML 2025. https://openreview.net/pdf?id=Jp988ELppQ
"""

from collections.abc import Callable
from typing import NamedTuple

import torch
from jaxtyping import Bool, Float, Int

from closed_form_discrete_diffusion_guidance_investigation.markov_chain import (
    MASK,
    MarkovChain,
)

RewardFn = Callable[[Int[torch.Tensor, "n seq_len"]], Float[torch.Tensor, " n"]]
"""Terminal reward, evaluated on complete sequences. May return ``-inf``."""

CompletionSampler = Callable[
    [Int[torch.Tensor, "n seq_len"], int, torch.Generator],
    Int[torch.Tensor, "n n_completions seq_len"],
]
"""Draws completions of partially masked sequences."""

IntermediateReward = Callable[
    [Float[torch.Tensor, "n n_completions"]], Float[torch.Tensor, " n"]
]
"""Reduces a particle's completion rewards to the score its potential is built from."""


class SteeringResult(NamedTuple):
    """Outcome of a batch of independent steering runs.

    Attributes:
        terminal_rewards: True reward of every particle at the end of each run.
        best_seen: Best *true* reward over everything a run generated -- its final
          particles and every completion drawn to score them.
        resample_fraction: Fraction of scheduled steps at which resampling occurred.
          Below one when ``ess_threshold`` is set, or when no particle in a run had a
          finite reward.
        non_finite_reward_fraction: Fraction of scheduled steps at which no particle in
          a run had a finite reward estimate, leaving nothing to select between. Such a
          particle is perfectly valid and still completes to a feasible sequence; it is
          only the completions drawn to score it that all came out infeasible.
    """

    terminal_rewards: Float[torch.Tensor, "n_runs n_particles"]
    best_seen: Float[torch.Tensor, " n_runs"]
    resample_fraction: float
    non_finite_reward_fraction: float


def monte_carlo_reward(
    *, over_feasible: bool = False, infeasible_reward: float | None = None
) -> IntermediateReward:
    """Build the log-mean-exp estimator with a chosen handling of infeasibility.

    The estimator of Singhal et al. is ``log (1/N) sum_j exp(r_j)``. Where the objective
    scores an infeasible completion ``-inf`` it contributes ``exp(-inf) = 0``, so with
    ``K`` the number of feasible completions the estimator decomposes exactly as

        log (1/N) sum_j exp(r_j) = log(K/N) + log (1/K) sum_{feasible} exp(r_j)

    -- how often the particle admits a valid completion at all, plus how good those
    completions are. The arguments give the two ways of stopping the first term from
    dominating, and passing neither reproduces the estimator as published.

    ``over_feasible`` divides by ``K`` instead of ``N``, dropping the first term.

    ``infeasible_reward`` instead replaces ``-inf`` with some supplied reward.

    Args:
        over_feasible: Normalise by the number of feasible completions, not by ``N``.
        infeasible_reward: Value substituted for a ``-inf`` completion reward.

    Returns:
        A reduction from per-completion rewards to one score per particle.
    """
    if over_feasible and infeasible_reward is not None:
        msg = (
            "over_feasible and infeasible_reward are mutually exclusive: flooring "
            "removes the infeasible completions that over_feasible needs to count."
        )
        raise ValueError(msg)

    def estimate(
        rewards: Float[torch.Tensor, "n n_completions"],
    ) -> Float[torch.Tensor, " n"]:
        n_particles, n_completions = rewards.shape
        if infeasible_reward is not None:
            rewards = torch.nan_to_num(rewards, neginf=infeasible_reward)
        if over_feasible:
            denominator = torch.isfinite(rewards).sum(dim=1).clamp(min=1).double()
        else:
            denominator = torch.full(
                (n_particles,), float(n_completions), dtype=torch.float64
            )
        return torch.logsumexp(rewards.double(), dim=1) - torch.log(denominator)

    return estimate


def _resampling_logits(
    reward: Float[torch.Tensor, "n_runs n_particles"],
    previous_reward: Float[torch.Tensor, "n_runs n_particles"],
    tilting_strength: float,
) -> tuple[
    Float[torch.Tensor, "n_runs n_particles"],
    Bool[torch.Tensor, " n_runs"],
    Float[torch.Tensor, "n_runs n_particles"],
]:
    """Log potentials for a step, which runs had no finite reward, and the carry.

    Uses the difference potential ``G_t = exp(lambda (r_t - r_t'))``, whose product
    telescopes to ``exp(lambda (r_last - r_first'))``.

    A particle whose reward estimate is ``-inf`` -- meaning every completion drawn to
    score it was infeasible -- gets a ``-inf`` logit, so it has zero weight and is not
    selected when resampling occurs. That alone does not keep ``previous_reward``
    finite, though: a run that is not due to resample, because its effective sample
    size is high or because none of its particles had a finite reward, keeps its
    population unchanged and those particles with it. The returned reward therefore
    substitutes zero for them explicitly.

    Where *no* particle in a run has a finite reward there is nothing to select between,
    so its logits are zeroed, leaving a uniform categorical; the caller leaves the run
    out of resampling rather than resampling from it. Every particle then carries zero,
    so the next step differences them all against the same baseline, which cancels.

    Returns:
        The log potential of each particle; the runs in which no particle had a finite
        reward, so that there is nothing to select between and resampling should be
        skipped; and the reward to difference against at the next scoring step, equal to
        ``reward`` except where a particle's reward was not finite.
    """
    finite_reward = torch.isfinite(reward)
    logits = (tilting_strength * (reward - previous_reward)).masked_fill(
        ~finite_reward, -float("inf")
    )
    # With nothing to select between, leave the population alone rather than resampling
    # from an all-`-inf` categorical.
    non_finite_reward = ~finite_reward.any(1)
    return (
        torch.where(non_finite_reward.unsqueeze(1), torch.zeros_like(logits), logits),
        non_finite_reward,
        torch.where(finite_reward, reward, torch.zeros_like(reward)),
    )


def fk_steer(
    chain: MarkovChain,
    reward_fn: RewardFn,
    completion_sampler: CompletionSampler,
    intermediate_reward: IntermediateReward,
    seq_len: int,
    n_particles: int,
    n_runs: int,
    n_completions: int,
    tilting_strength: float,
    generator: torch.Generator,
    resample_every: int = 4,
    ess_threshold: float | None = None,
    guidance_reward_fn: RewardFn | None = None,
) -> SteeringResult:
    """Steer a population of particles from fully masked to complete.

    ``n_runs`` independent populations are advanced together as one batch.

    Args:
        chain: Chain being denoised. Supplies the exact single-token proposal.
        reward_fn: Terminal reward, evaluated on complete sequences.
        completion_sampler: Draws the completions used to score partially masked
          particles. The chain's joint sampler gives exact intermediate rewards; its
          marginal sampler gives the factorised approximation a trained model makes.
        intermediate_reward: Reduces those completions' rewards to one score per
          particle; see :func:`monte_carlo_reward`.
        seq_len: Length of the sequences to generate.
        n_particles: Particles per run.
        n_runs: Independent populations to advance in parallel.
        n_completions: Completions drawn per particle at each scoring step.
        tilting_strength: ``lambda`` in the potential.
        generator: Torch generator, for reproducibility.
        resample_every: Score and resample every this many committed positions.
        ess_threshold: If given, resample only where the effective sample size is below
          ``ess_threshold * n_particles``.
        guidance_reward_fn: Reward the potentials are built from, if it differs from the
          one the run is judged by. Defaults to ``reward_fn``. Note that if this is
          passed then ``terminal_rewards`` and ``best_seen`` are still evaluated using
          the true `reward_fn`.

    Returns:
        A :class:`SteeringResult` for the batch.
    """
    guides = guidance_reward_fn if guidance_reward_fn is not None else reward_fn
    device = chain.transition_matrix.device
    total = n_runs * n_particles
    x_t = torch.full((total, seq_len), MASK, dtype=torch.long, device=device)
    previous_reward = torch.zeros(
        n_runs, n_particles, dtype=torch.float64, device=device
    )
    best_seen = torch.full((n_runs,), -float("inf"), dtype=torch.float64, device=device)
    scheduled = resampled = non_finite_reward_runs = 0

    for step in range(1, seq_len + 1):
        # Commit exactly one position per particle, from its conditional given
        # everything already committed.
        position = (
            torch
            .rand(total, seq_len, generator=generator, device=device)
            .masked_fill(x_t != MASK, -1.0)
            .argmax(1, keepdim=True)
        )
        drawn = chain.sample_x0_given_xt_via_marginals(x_t, 1, generator).squeeze(1)
        x_t = x_t.scatter(1, position, drawn.gather(1, position))

        if tilting_strength <= 0 or step % resample_every or step == seq_len:
            continue

        scored = completion_sampler(x_t, n_completions, generator).reshape(
            -1, seq_len
        )  # [total * n_completions, seq_len]

        raw = guides(scored).double().view(total, n_completions)
        # `best_seen` is a claim about what the run found, so it is always the true
        # reward.
        raw_true = (
            raw
            if guidance_reward_fn is None
            else reward_fn(scored).double().view(total, n_completions)
        )
        best_seen = torch.maximum(best_seen, raw_true.view(n_runs, -1).max(1).values)
        reward = intermediate_reward(raw).view(n_runs, n_particles)

        logits, non_finite_reward, previous_reward = _resampling_logits(
            reward, previous_reward, tilting_strength
        )  # ([n_runs, n_particles], [n_runs], [n_runs, n_particles])
        weights = torch.softmax(logits, dim=1)

        scheduled += n_runs
        non_finite_reward_runs += int(non_finite_reward.sum())
        if ess_threshold is None:
            due = ~non_finite_reward
        else:
            due = (
                (1.0 / (weights**2).sum(1)) < ess_threshold * n_particles
            ) & ~non_finite_reward
        resampled += int(due.sum())
        if not bool(due.any()):
            continue

        index = torch.where(
            due.unsqueeze(1),
            torch.multinomial(
                weights, n_particles, replacement=True, generator=generator
            ),
            torch.arange(n_particles, device=device),
        )  # [n_runs, n_particles]
        x_t = (
            x_t
            .view(n_runs, n_particles, seq_len)
            .gather(1, index.unsqueeze(-1).expand(-1, -1, seq_len))
            .view(total, seq_len)
        )
        previous_reward = previous_reward.gather(1, index)

    terminal = reward_fn(x_t).double().view(n_runs, n_particles)
    return SteeringResult(
        terminal_rewards=terminal,
        best_seen=torch.maximum(best_seen, terminal.max(1).values),
        resample_fraction=resampled / scheduled if scheduled else 1.0,
        non_finite_reward_fraction=(
            non_finite_reward_runs / scheduled if scheduled else 0.0
        ),
    )


def best_of_n(
    chain: MarkovChain,
    reward_fn: RewardFn,
    seq_len: int,
    n: int,
    n_groups: int,
    generator: torch.Generator,
    groups_per_batch: int = 25,
) -> Float[torch.Tensor, " n_groups"]:
    """Best reward from ``n`` unguided draws, over ``n_groups`` independent repeats.

    Args:
        chain: Chain to draw unguided samples from.
        reward_fn: Terminal reward, evaluated on complete sequences.
        seq_len: Length of the sequences to draw.
        n: Samples drawn per group.
        n_groups: Independent repeats, for an error bar.
        generator: Torch generator, for reproducibility.
        groups_per_batch: Groups drawn per call to the sampler, to bound peak memory.

    Returns:
        The maximum reward within each group.
    """
    best = []
    remaining = n_groups
    while remaining > 0:
        batch = min(groups_per_batch, remaining)
        samples = chain.sample_full(seq_len, batch * n, generator)
        best.append(reward_fn(samples).double().view(batch, n).max(dim=1).values)
        remaining -= batch
    return torch.cat(best)
