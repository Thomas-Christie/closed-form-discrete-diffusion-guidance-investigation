"""Tests for :class:`MarkovChain`, against a brute-force enumeration of the space.

The chain under test is the one implied by an :class:`Ehrlich` problem: a sequence is a
feasible Ehrlich input exactly when it has non-zero likelihood under this chain.

With ``num_states=6, dim=7`` there are only 279,936 sequences, so the entire space can
be enumerated and scored exactly, and conditioning on a partially observed ``x_t`` is
just subsetting that table and renormalising.

Two structural facts drive the conditional-sampling tests:

* Both conditional samplers run the same forward-backward pass, so both reproduce the
  exact per-position *marginals*.
* They differ in their *joint* structure. The cheapest statistic with real power is the
  distribution over adjacent masked pairs -- only ``K ** 2`` bins, so the Monte Carlo
  noise floor is small, and it is exactly what a factorised sampler gets wrong. The
  full joint is checkable as well.
"""

import itertools

import pytest
import torch
from holo.test_functions.closed_form import Ehrlich
from jaxtyping import Float, Int

from closed_form_discrete_diffusion_guidance_investigation.markov_chain import (
    MASK,
    MarkovChain,
)

# `num_states` controls sparsity via holo's `bandwidth = int(num_states * 0.4)`, and it
# is not monotone: at `num_states=5` the bandwidth rounds down far enough that the
# transition matrix comes out fully dense, every sequence is feasible, and the manifold
# constraint disappears entirely. Six is the smallest value that keeps it meaningful --
# `test_transition_matrix_is_sparse` guards this.
NUM_STATES = 6
DIM = 7
NUM_MOTIFS = 1
MOTIF_LENGTH = 3

N_SAMPLES = 100_000
SEED = 42

# Number of sequences checked against the deliberately slow per-sequence reference.
N_NAIVE_REFERENCE = 500


def mc_noise_floor(n_bins: int, n_samples: int) -> float:
    """Expected total-variation error from Monte Carlo alone, for calibration.

    Asymptotic under a uniform distribution, which is the worst case; concentrated or
    sparsely-supported distributions come in below it. Used only to scale tolerances.

    Args:
        n_bins: Number of distinct outcomes.
        n_samples: Number of samples drawn.

    Returns:
        Expected total-variation distance between the empirical and true distribution.
    """
    return (n_bins / (2 * torch.pi * n_samples)) ** 0.5


@pytest.fixture(scope="module")
def ehrlich() -> Ehrlich:
    """The Ehrlich problem whose validity manifold is under test."""
    return Ehrlich(
        num_states=NUM_STATES,
        dim=DIM,
        num_motifs=NUM_MOTIFS,
        motif_length=MOTIF_LENGTH,
        random_seed=SEED,
    )


@pytest.fixture(scope="module")
def chain(ehrlich: Ehrlich) -> MarkovChain:
    """The chain implied by the Ehrlich problem."""
    return MarkovChain(
        initial_dist=ehrlich.stationary_dist.squeeze(-1),
        transition_matrix=ehrlich.transition_matrix,
    )


@pytest.fixture(scope="module")
def seqs() -> Int[torch.Tensor, "n_seqs dim"]:
    """Every sequence in the space."""
    return torch.tensor(list(itertools.product(range(NUM_STATES), repeat=DIM)))


@pytest.fixture(scope="module")
def probs(
    chain: MarkovChain, seqs: Int[torch.Tensor, "n_seqs dim"]
) -> Float[torch.Tensor, " n_seqs"]:
    """Exact probability of every sequence, scored by the implementation under test."""
    return chain.log_prob(seqs).exp()


@pytest.fixture(scope="module")
def incumbent(chain: MarkovChain, ehrlich: Ehrlich) -> Int[torch.Tensor, " dim"]:
    """A known-feasible incumbent to apply mask patterns to."""
    seq = ehrlich.initial_solution(1).long()
    assert torch.isfinite(chain.log_prob(seq.unsqueeze(0))).all()
    return seq


@pytest.fixture(scope="module")
def test_cases(
    incumbent: Int[torch.Tensor, " dim"],
) -> dict[str, Int[torch.Tensor, " dim"]]:
    """Masked sequences, keyed by the structure each one exercises.

    Whether masked positions are *contiguous* determines whether the factorised sampler
    is exact.
    """
    x = incumbent
    return {
        "first + last observed": torch.tensor([
            x[0],
            MASK,
            MASK,
            MASK,
            MASK,
            MASK,
            x[6],
        ]),
        "+ middle observed": torch.tensor([x[0], MASK, MASK, x[3], MASK, MASK, x[6]]),
        "only last observed": torch.tensor([MASK, MASK, MASK, MASK, MASK, MASK, x[6]]),
        "only first observed": torch.tensor([x[0], MASK, MASK, MASK, MASK, MASK, MASK]),
        "isolated masks (null)": torch.tensor([
            x[0],
            MASK,
            x[2],
            MASK,
            x[4],
            x[5],
            x[6],
        ]),
        "fully masked": torch.full((DIM,), MASK),
    }


CASE_NAMES = [
    "first + last observed",
    "+ middle observed",
    "only last observed",
    "only first observed",
    "isolated masks (null)",
    "fully masked",
]
ADJACENT_CASES = [n for n in CASE_NAMES if n != "isolated masks (null)"]
ISOLATED_CASES = ["isolated masks (null)"]


def exact_posterior(
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
    x_t: Int[torch.Tensor, " dim"],
) -> Float[torch.Tensor, " n_seqs"]:
    """Exact ``q(x_0 | x_t)``, by subsetting the enumerated table and renormalising.

    Infeasible sequences already carry probability zero, so they drop out of any
    probability-weighted sum without needing to be filtered explicitly.
    """
    keep = torch.ones(seqs.shape[0], dtype=torch.bool)
    for i in range(DIM):
        if x_t[i] != MASK:
            keep &= seqs[:, i] == x_t[i]
    masked_probs = probs * keep
    assert masked_probs.sum() > 0, "x_t admits no feasible completion"
    return masked_probs / masked_probs.sum()


def block_distribution(
    seqs: Int[torch.Tensor, "n dim"],
    positions: list[int],
    weights: Float[torch.Tensor, " n"] | None = None,
) -> Float[torch.Tensor, " n_outcomes"]:
    """Joint distribution over the ``K ** len(positions)`` outcomes at `positions`.

    Pass `weights` to reweight the enumerated table; omit it to histogram samples.
    A single position gives that position's marginal; a pair gives their joint. Both
    sides of a comparison therefore go through identical binning and cannot disagree
    on convention.

    Args:
        seqs: Sequences to reduce.
        positions: Positions whose joint distribution is wanted.
        weights: Per-sequence probabilities, or ``None`` to count occurrences.

    Returns:
        A distribution over ``K ** len(positions)`` outcomes.
    """
    key = torch.zeros(seqs.shape[0], dtype=torch.long)
    for i in positions:
        key = key * NUM_STATES + seqs[:, i]
    n_outcomes = NUM_STATES ** len(positions)
    if weights is None:
        return torch.bincount(key, minlength=n_outcomes).double() / seqs.shape[0]
    return torch.zeros(n_outcomes, dtype=weights.dtype).index_add_(0, key, weights)


# --------------------------------------------------------------------------------
# The problem itself
# --------------------------------------------------------------------------------


def test_transition_matrix_is_sparse(chain: MarkovChain) -> None:
    """Some bigrams must be forbidden, or the manifold constraint is vacuous.

    holo derives sparsity from ``bandwidth = int(num_states * 0.4)``, which for small
    ``num_states`` can round down to a fully dense matrix. Every feasibility assertion
    below would then pass trivially.
    """
    assert (chain.transition_matrix == 0).any(), "transition matrix is dense"
    assert torch.allclose(
        chain.transition_matrix.sum(-1), torch.ones(NUM_STATES, dtype=torch.float64)
    ), "transition matrix is not row-stochastic"


# --------------------------------------------------------------------------------
# log_prob
# --------------------------------------------------------------------------------


def test_log_prob_is_normalised(probs: Float[torch.Tensor, " n_seqs"]) -> None:
    """The induced distribution over the whole space sums to one."""
    assert torch.isclose(probs.sum(), torch.ones((), dtype=torch.float64))


def test_log_prob_matches_naive_reference(
    chain: MarkovChain, seqs: Int[torch.Tensor, "n_seqs dim"]
) -> None:
    """`log_prob` agrees with a deliberately slow, obviously-correct reference."""
    index = torch.randperm(
        seqs.shape[0], generator=torch.Generator().manual_seed(SEED)
    )[:N_NAIVE_REFERENCE]

    naive = []
    for seq in seqs[index]:
        lp = chain.log_initial_dist[seq[0]]
        for i in range(1, DIM):
            lp = lp + chain.log_transition_matrix[seq[i - 1], seq[i]]
        naive.append(lp)

    assert torch.allclose(chain.log_prob(seqs[index]), torch.stack(naive))


def test_log_prob_is_independent_of_batch(
    chain: MarkovChain, seqs: Int[torch.Tensor, "n_seqs dim"]
) -> None:
    """A sequence scores the same alone as it does alongside others."""
    subset = seqs[:: seqs.shape[0] // 32]
    one_at_a_time = torch.cat([chain.log_prob(s.unsqueeze(0)) for s in subset])
    assert torch.allclose(one_at_a_time, chain.log_prob(subset))


def test_log_prob_is_neg_inf_exactly_on_forbidden_bigrams(
    chain: MarkovChain, seqs: Int[torch.Tensor, "n_seqs dim"]
) -> None:
    """Infeasibility is reported for exactly the sequences containing a zero bigram."""
    forbidden = (chain.transition_matrix[seqs[:, :-1], seqs[:, 1:]] == 0).any(-1)
    assert torch.equal(torch.isneginf(chain.log_prob(seqs)), forbidden)


def test_feasibility_agrees_with_the_ehrlich_objective(
    chain: MarkovChain, ehrlich: Ehrlich, seqs: Int[torch.Tensor, "n_seqs dim"]
) -> None:
    """The chain's notion of validity is the objective's."""
    objective = ehrlich(seqs.float())
    assert torch.equal(torch.isneginf(chain.log_prob(seqs)), torch.isneginf(objective))


# --------------------------------------------------------------------------------
# check_feasibility
# --------------------------------------------------------------------------------


def greedy_walk(chain: MarkovChain, start: int, length: int) -> list[int]:
    """A feasible sequence of `length` tokens beginning at `start`."""
    seq = [start]
    while len(seq) < length:
        seq.append(int(chain.transition_matrix[seq[-1]].argmax()))
    return seq


def test_check_feasibility_on_hand_built_sequences(chain: MarkovChain) -> None:
    """A deliberately constructed feasible and infeasible sequence are told apart.

    Both sequences start from a state that has *some* unreachable successor, which
    `test_transition_matrix_is_sparse` guarantees exists. The feasible one then walks
    greedily; the infeasible one takes the unreachable step first and walks greedily
    afterwards, so its only forbidden bigram is the deliberate one at position 0 -> 1.
    Nothing here depends on where the sparsity happens to fall.
    """
    start = int((chain.transition_matrix == 0).any(-1).nonzero()[0])
    unreachable = int((chain.transition_matrix[start] == 0).nonzero()[0])
    assert chain.initial_dist[start] > 0, "the walk must start from a possible token"

    feasible = greedy_walk(chain, start, DIM)
    infeasible = [start, *greedy_walk(chain, unreachable, DIM - 1)]

    seqs = torch.tensor([feasible, infeasible])
    assert torch.equal(chain.check_feasibility(seqs), torch.tensor([True, False])), seqs


def test_check_feasibility_rejects_an_impossible_first_token() -> None:
    """A first token of zero probability is infeasible even with every bigram allowed.

    The Ehrlich chain starts from a strictly positive stationary distribution, so
    neither the enumerated space nor the test above can exercise the initial-
    distribution branch at all. This builds a chain with a hole in its initial
    distribution and a fully dense transition matrix, isolating that branch.
    """
    chain = MarkovChain(
        initial_dist=torch.tensor([0.5, 0.5, 0.0]),
        transition_matrix=torch.full((3, 3), 1 / 3),
    )
    seqs = torch.tensor([[0, 2, 1], [2, 0, 1]])
    assert torch.equal(chain.check_feasibility(seqs), torch.tensor([True, False]))


def test_check_feasibility_matches_log_prob(
    chain: MarkovChain,
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
) -> None:
    """Over the whole enumerated space, feasibility is exactly non-zero likelihood.

    ``log_prob`` is independently validated above, so this pins ``check_feasibility``
    to it across all 279,936 sequences rather than the two hand-built cases -- and
    would catch a check that dropped the initial distribution or the final bigram.
    """
    assert torch.equal(chain.check_feasibility(seqs), probs > 0)


# --------------------------------------------------------------------------------
# sample_full
# --------------------------------------------------------------------------------


def test_sample_full_returns_feasible_sequences(chain: MarkovChain) -> None:
    """Ancestral sampling never leaves the manifold."""
    samples = chain.sample_full(DIM, N_SAMPLES, torch.Generator().manual_seed(SEED))
    assert torch.isfinite(chain.log_prob(samples)).all()


def test_sample_full_matches_the_exact_marginals(
    chain: MarkovChain,
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
) -> None:
    """Every per-position marginal of the unconditional sampler matches the chain."""
    samples = chain.sample_full(DIM, N_SAMPLES, torch.Generator().manual_seed(SEED))
    tolerance = 3 * mc_noise_floor(NUM_STATES, N_SAMPLES)
    for i in range(DIM):
        empirical = block_distribution(samples, [i])
        exact = block_distribution(seqs, [i], probs)
        assert (empirical - exact).abs().max() < tolerance, i


def test_sample_full_gets_the_first_bigram_right(chain: MarkovChain) -> None:
    """Position 0 must be *coupled* to position 1, not merely marginally correct."""
    samples = chain.sample_full(DIM, N_SAMPLES, torch.Generator().manual_seed(SEED))
    empirical = block_distribution(samples, [0, 1]).reshape(NUM_STATES, NUM_STATES)
    exact = chain.initial_dist.unsqueeze(-1) * chain.transition_matrix
    tolerance = 2 * mc_noise_floor(NUM_STATES**2, N_SAMPLES)
    assert (empirical - exact).abs().max() < tolerance


# --------------------------------------------------------------------------------
# Conditional sampling
# --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drawn(
    chain: MarkovChain, test_cases: dict[str, Int[torch.Tensor, " dim"]]
) -> dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]]:
    """Samples from both conditional samplers, for every test case."""
    generator = torch.Generator().manual_seed(SEED)
    names = list(test_cases)
    x_t = torch.stack([test_cases[name] for name in names])
    joint = chain.sample_x0_given_xt_via_joint(x_t, N_SAMPLES, generator)
    marginals = chain.sample_x0_given_xt_via_marginals(x_t, N_SAMPLES, generator)
    return {
        name: {"joint": joint[i], "marginals": marginals[i]}
        for i, name in enumerate(names)
    }


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("sampler", ["joint", "marginals"])
def test_observed_positions_are_preserved(
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    test_cases: dict[str, Int[torch.Tensor, " dim"]],
    case: str,
    sampler: str,
) -> None:
    """Neither sampler may alter a position that was observed."""
    x_t = test_cases[case]
    observed = x_t != MASK
    samples = drawn[case][sampler]
    assert torch.equal(samples[:, observed], x_t[observed].expand(N_SAMPLES, -1))


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("sampler", ["joint", "marginals"])
def test_both_samplers_reproduce_the_exact_marginals(
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    test_cases: dict[str, Int[torch.Tensor, " dim"]],
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
    case: str,
    sampler: str,
) -> None:
    """Both samplers get every per-position marginal right: this is forward-backward."""
    x_t = test_cases[case]
    posterior = exact_posterior(seqs, probs, x_t)
    tolerance = 2.0 * mc_noise_floor(NUM_STATES, N_SAMPLES)
    for i in (x_t == MASK).nonzero().squeeze(-1).tolist():
        empirical = block_distribution(drawn[case][sampler], [i])
        exact = block_distribution(seqs, [i], posterior)
        assert (empirical - exact).abs().max() < tolerance, (case, sampler, i)


@pytest.mark.parametrize("case", CASE_NAMES)
def test_joint_sampler_stays_on_the_manifold(
    chain: MarkovChain,
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    case: str,
) -> None:
    """Exact joint sampling can only ever produce feasible sequences."""
    assert torch.isfinite(chain.log_prob(drawn[case]["joint"])).all()


@pytest.mark.parametrize("case", ADJACENT_CASES)
def test_joint_sampler_gets_adjacent_pairs_right(
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    test_cases: dict[str, Int[torch.Tensor, " dim"]],
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
    case: str,
) -> None:
    """The joint sampler reproduces the dependence between adjacent masked positions."""
    x_t = test_cases[case]
    posterior = exact_posterior(seqs, probs, x_t)
    masked = (x_t == MASK).nonzero().squeeze(-1).tolist()
    tolerance = mc_noise_floor(NUM_STATES**2, N_SAMPLES)
    for i in masked:
        if i + 1 not in masked:
            continue
        empirical = block_distribution(drawn[case]["joint"], [i, i + 1])
        exact = block_distribution(seqs, [i, i + 1], posterior)
        assert (empirical - exact).abs().max() < tolerance, (case, i)


@pytest.mark.parametrize("case", CASE_NAMES)
def test_joint_sampler_reproduces_the_full_joint(
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    test_cases: dict[str, Int[torch.Tensor, " dim"]],
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
    case: str,
) -> None:
    """The joint sampler reproduces the entire posterior, not just adjacent pairs."""
    x_t = test_cases[case]
    posterior = exact_posterior(seqs, probs, x_t)
    masked = (x_t == MASK).nonzero().squeeze(-1).tolist()
    exact = block_distribution(seqs, masked, posterior)
    empirical = block_distribution(drawn[case]["joint"], masked)
    tolerance = mc_noise_floor(int((exact > 0).sum()), N_SAMPLES)
    total_variation = 0.5 * (empirical - exact).abs().sum()
    assert total_variation < tolerance, (case, total_variation.item(), tolerance)


@pytest.mark.parametrize("case", ISOLATED_CASES)
def test_marginal_samples_are_feasible_when_masks_are_isolated(
    chain: MarkovChain,
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    case: str,
) -> None:
    """With no two masked positions adjacent, the joint *is* the product of marginals."""  # ruff: ignore[line-too-long]
    assert torch.isfinite(chain.log_prob(drawn[case]["marginals"])).all()


@pytest.mark.parametrize("case", ADJACENT_CASES)
def test_marginal_sampler_gets_adjacent_pairs_wrong(
    chain: MarkovChain,
    drawn: dict[str, dict[str, Int[torch.Tensor, "n_samples dim"]]],
    test_cases: dict[str, Int[torch.Tensor, " dim"]],
    seqs: Int[torch.Tensor, "n_seqs dim"],
    probs: Float[torch.Tensor, " n_seqs"],
    case: str,
) -> None:
    """Contiguous masked positions expose the factorisation error."""
    x_t = test_cases[case]
    posterior = exact_posterior(seqs, probs, x_t)
    samples = drawn[case]["marginals"]
    masked = (x_t == MASK).nonzero().squeeze(-1).tolist()
    floor = mc_noise_floor(NUM_STATES**2, N_SAMPLES)

    worst = max(
        (
            block_distribution(samples, [i, i + 1])
            - block_distribution(seqs, [i, i + 1], posterior)
        )
        .abs()
        .max()
        .item()
        for i in masked
        if i + 1 in masked
    )
    assert worst > 3 * floor, (case, worst)
    assert not torch.isfinite(chain.log_prob(samples)).all(), case


def test_joint_sampler_hits_a_two_sequence_posterior_exactly() -> None:
    """Where the posterior has two completions, every draw is one of them.

    An assertion with no Monte Carlo tolerance at all, which the Ehrlich posterior is
    far too broad to support. The two lanes ``1 -> 2`` and ``2 -> 1`` are the only
    bridges between the observed endpoints, so each masked position is uniform on
    ``{1, 2}`` while the pair is not.
    """
    chain = MarkovChain(
        initial_dist=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        transition_matrix=torch.tensor([
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0],
        ]),
    )
    x_t = torch.tensor([[0, MASK, MASK, 3]])
    generator = torch.Generator().manual_seed(SEED)

    samples = chain.sample_x0_given_xt_via_joint(x_t, N_SAMPLES, generator)[0]
    expected = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]])
    assert torch.equal(torch.unique(samples, dim=0), expected)

    factorised = chain.sample_x0_given_xt_via_marginals(x_t, N_SAMPLES, generator)[0]
    assert len(torch.unique(factorised, dim=0)) == 4
