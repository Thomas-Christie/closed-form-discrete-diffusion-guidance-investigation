# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: closed-form-discrete-diffusion-guidance-investigation (3.13.5)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiments
#
# Three experiments:
#
# 1. **Feasibility** - how often a completion of a partially masked sequence is feasible, as a function of how much is masked.
# 2. **Reward agreement** - how well each factorised intermediate-reward estimator
#    ranks particles relative to the exact joint estimate.
# 3. **FK steering** - what that ranking error costs when the estimator is used to
#    steer.
#
# Note - experiment results can be found in `notebooks/results/paper_results.pt`, and can be used to generate the figures for the paper using the `Figures` section of this notebook directly, without needing to run the experiments.

# %% [markdown]
# ## Setup

# %%
import math
from dataclasses import dataclass
from pathlib import Path

import jaxtyping
import matplotlib.pyplot as plt
import torch
from holo.test_functions.closed_form import Ehrlich
from holo.test_functions.closed_form._ehrlich import _cubic_response  # noqa: PLC2701
from holo.test_functions.elemental import motif_search
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from tueplots import bundles
from tueplots.constants.color import palettes

plt.rcParams.update({"figure.dpi": 200})

jaxtyping.install_import_hook("closed_form_discrete_diffusion_guidance_investigation", "beartype.beartype")

# %load_ext jaxtyping
# %jaxtyping.typechecker beartype.beartype

from closed_form_discrete_diffusion_guidance_investigation.markov_chain import MarkovChain
from closed_form_discrete_diffusion_guidance_investigation.markov_chain import MASK
from closed_form_discrete_diffusion_guidance_investigation.fk_steering import (
    IntermediateReward,
    RewardFn,
    best_of_n,
    fk_steer,
    monte_carlo_reward,
)
from closed_form_discrete_diffusion_guidance_investigation.utils import seed_everything

seed_everything(42)

# %%
QUANTIZATION = 4

RESULTS_PATH = Path("results/paper_results.pt")
FIGURE_DIR = Path("figures")

SEEDS = tuple(range(10))
"""One Ehrlich function per seed. The figures generally report the mean over these
seeds and its standard error."""

# The experiment generators are seeded `<base> + problem.problem_seed`, which the two
# families share.
POOL_BASE, FEASIBILITY_BASE, ESTIMATE_BASE, STEERING_BASE = 1000, 2000, 3000, 4000
POOL_SEEDS_PER_FAMILY = 100
"""Pool seeds set aside for each family, and so the most seeds `SEEDS` can hold."""

if len(SEEDS) > POOL_SEEDS_PER_FAMILY:  # two families would share a pool seed
    message = (
        f"SEEDS is longer than POOL_SEEDS_PER_FAMILY "
        f"({len(SEEDS)} > {POOL_SEEDS_PER_FAMILY})"
    )
    raise ValueError(message)

# Mask levels at which the reward estimators are compared. The scatter figure shows a
# subset of them, so that both figures come out of one sweep.
MASK_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
SCATTER_FRACTIONS = (0.25, 0.5, 0.75)
N_PARTICLES_ANALYSIS, N_COMPLETIONS_ANALYSIS = 1000, 100
N_SAMPLES = 10000

N_PARTICLES, N_COMPLETIONS, RESAMPLE_EVERY, ESS_THRESHOLD = 16, 16, 4, None
N_RUNS = 100
TILTING_STRENGTHS = (10.0, 100.0, 1000.0, 5000.0, 20000.0, 100000.0)
BASELINE_LABEL = f"Unguided best-of-{N_PARTICLES}"


# %%
@dataclass(frozen=True)
class Arm:
    """One way of estimating a particle's intermediate reward.

    Every arm but the first replaces the exact joint conditional with the factorised
    one; they differ only in how they treat a completion that comes out infeasible,
    which the objective scores `-inf`.

    Attributes:
        name: Key used in the results file.
        colour: Line colour, shared across every figure.
        dash: Line style.
        label: One-line legend entry.
        joint_sampler: Draw completions from the joint conditional, not the marginals.
        estimator: Reduces a particle's completion rewards to one score.
        motif_guidance: Score completions with the always-finite motif reward instead
          of the objective, so infeasibility never produces a `-inf` to handle.
    """

    name: str
    colour: str
    dash: str
    label: str
    joint_sampler: bool
    estimator: IntermediateReward
    motif_guidance: bool


JOINT, FACTORISED_OVER_N, FACTORISED_OVER_K, FACTORISED_FLOORED, FACTORISED_MOTIF = (
    f"#{hex_code}" for hex_code in palettes.paultol_bright[:5]
)

ARMS = (
    Arm(
        name="joint",
        colour=JOINT,
        dash="-",
        label="Joint",
        joint_sampler=True,
        estimator=monte_carlo_reward(),
        motif_guidance=False,
    ),
    Arm(
        name="factorised_over_n",
        colour=FACTORISED_OVER_N,
        dash="-",
        label=r"Factorised, $r(\mathbf{x}_0) = -\infty$ if infeasible",
        joint_sampler=False,
        estimator=monte_carlo_reward(),
        motif_guidance=False,
    ),
    Arm(
        name="factorised_over_k",
        colour=FACTORISED_OVER_K,
        dash="--",
        label=r"Factorised, $\hat r(\mathbf{x}_t^{(i)})$ on feasible only",
        joint_sampler=False,
        estimator=monte_carlo_reward(over_feasible=True),
        motif_guidance=False,
    ),
    Arm(
        name="factorised_floored",
        colour=FACTORISED_FLOORED,
        dash="--",
        label=r"Factorised, $r(\mathbf{x}_0) = 0$ if infeasible",
        joint_sampler=False,
        estimator=monte_carlo_reward(infeasible_reward=0.0),
        motif_guidance=False,
    ),
    Arm(
        name="factorised_motif",
        colour=FACTORISED_MOTIF,
        dash=":",
        label="Factorised, motif reward",
        joint_sampler=False,
        estimator=monte_carlo_reward(),
        motif_guidance=True,
    ),
)

# In the agreement experiments the joint arm is a *second*, independent joint draw
# scored against the first, so its curve is the Monte-Carlo noise floor rather than a
# method.
ANALYSIS_LABELS = {"joint": "Independent joint draw (control)"}


def legend_label(arm: Arm, *, analysis: bool = False) -> str:
    """Legend entry for `arm`, named for the role it plays in that figure."""
    return ANALYSIS_LABELS.get(arm.name, arm.label) if analysis else arm.label


# %%
@dataclass(frozen=True)
class Family:
    """One Ehrlich configuration, instantiated once per seed in `SEEDS`.

    Attributes:
        name: Key used in the results file, shared by every instance of the family.
        label: Name used in figures.
        dim: Sequence length.
    """

    name: str
    label: str
    dim: int


FAMILIES = (Family("short", "Ehr-Short", dim=32), Family("long", "Ehr-Long", dim=128))


@dataclass(frozen=True, eq=False)
class Problem:
    """One Ehrlich instance, its chain, and a pool of feasible sequences to mask.

    Attributes:
        family: The configuration this is an instance of.
        problem_seed: Draws the instance, and seeds the experiments run on it.
        ehrlich: The objective.
        chain: The chain whose support the objective is defined on.
        samples: Sequences drawn from the chain.
    """

    family: Family
    problem_seed: int
    ehrlich: Ehrlich
    chain: MarkovChain
    samples: Int[torch.Tensor, "n_seqs seq_len"]

    @property
    def name(self) -> str:
        """Key used in the results file."""
        return self.family.name

    @property
    def label(self) -> str:
        """Name used in figures."""
        return self.family.label

    @property
    def seq_len(self) -> int:
        """Length of the sequences this problem is defined over."""
        return self.ehrlich.dim


def make_problem(
    family: Family, problem_seed: int, sample_seed: int, n_samples: int
) -> Problem:
    """Build one instance of `family`, with a pool of feasible draws.

    `problem_seed` fixes the objective and the chain; `sample_seed` fixes the pool drawn
    from that chain.
    """
    ehrlich = Ehrlich(
        num_states=32,
        dim=family.dim,
        num_motifs=4,
        motif_length=4,
        quantization=QUANTIZATION,
        random_seed=problem_seed,
    )
    chain = MarkovChain(
        initial_dist=ehrlich.initial_dist, transition_matrix=ehrlich.transition_matrix
    )
    samples = chain.sample_full(
        family.dim,
        n_samples,
        torch.Generator(device="cpu").manual_seed(sample_seed),
    )
    return Problem(
        family=family,
        problem_seed=problem_seed,
        ehrlich=ehrlich,
        chain=chain,
        samples=samples,
    )


REPLICATES = {
    family.name: tuple(
        make_problem(
            family,
            problem_seed=seed,
            sample_seed=POOL_BASE + POOL_SEEDS_PER_FAMILY * index + seed,
            n_samples=N_SAMPLES,
        )
        for seed in SEEDS
    )
    for index, family in enumerate(FAMILIES)
}
"""Every family's instances, in `SEEDS` order."""


# %%
def true_reward(ehrlich: Ehrlich) -> RewardFn:
    """The Ehrlich objective itself."""
    return lambda seqs: ehrlich(seqs.float())


def motif_reward(
    ehrlich: Ehrlich,
    seqs: Float[torch.Tensor, "n seq_len"],
    epistasis_factor: float = 0.0,
) -> Float[torch.Tensor, " n"]:
    """Ehrlich's motif score with the feasibility mask removed.

    `Ehrlich._evaluate_true` multiplies its per-motif contributions together and then
    replaces the result with `-inf` wherever the sequence has zero likelihood under the
    chain. This reproduces everything up to that replacement, so it agrees exactly with
    the objective on feasible sequences and stays finite for infeasible sequences.
    """
    present = [
        motif_search(
            solution=seqs,
            motif=motif,
            spacing=spacing,
            mode="present",
            quantization=QUANTIZATION,
        )
        for motif, spacing in zip(ehrlich.motifs, ehrlich.spacings, strict=True)
    ]
    return torch.stack([
        _cubic_response(count, epistasis_factor) for count in present
    ]).prod(dim=0)


def guidance_reward(arm: Arm, ehrlich: Ehrlich) -> RewardFn | None:
    """Reward the arm's potentials are built from, or `None` to use the objective."""
    if not arm.motif_guidance:
        return None
    return lambda seqs: motif_reward(ehrlich, seqs.float())


# %%
# `motif_reward` reconstructs a modified form of the objective rather than calling it,
# so `QUANTIZATION` and the epistasis factor have to match the instance. Check that it
# agrees with the objective where the objective is finite, and stays finite where it is
# not -- on every instance, since each has its own motifs and spacings.
for family in FAMILIES:
    for problem in REPLICATES[family.name]:
        generator = torch.Generator(device="cpu").manual_seed(53)
        feasible = problem.chain.sample_full(problem.seq_len, 2000, generator).float()
        off_manifold = torch.randint(  # infeasible with high probability
            problem.ehrlich.num_states, (2000, problem.seq_len), generator=generator
        ).float()
        if not torch.equal(
            problem.ehrlich(feasible).double(),
            motif_reward(problem.ehrlich, feasible).double(),
        ):
            message = (
                f"motif_reward disagrees with the objective on {problem.label} "
                f"seed {problem.problem_seed}; QUANTIZATION or the epistasis "
                "factor does not match the instance"
            )
            raise ValueError(message)
        instance = f"{problem.label} seed {problem.problem_seed}:"
        motif_finite = torch.isfinite(motif_reward(problem.ehrlich, off_manifold)).all()
        objective_finite = torch.isfinite(problem.ehrlich(off_manifold)).all()
        if not motif_finite:
            message = (
                f"motif_reward returns -inf on an off-manifold row of {problem.label} "
                f"seed {problem.problem_seed}; the motif arm's potentials rely on "
                "it staying finite wherever the objective does not"
            )
            raise ValueError(message)
        print(
            f"{instance:<17} motif_reward matches the objective on 2000 feasible "
            f"sequences; off-manifold rows finite under motif_reward "
            f"{motif_finite.item()}, under the objective {objective_finite.item()}"
        )


# %%
def mask_positions(
    samples: Int[torch.Tensor, "n_seqs seq_len"],
    n_masked: int,
    generator: torch.Generator,
) -> Int[torch.Tensor, "n_seqs seq_len"]:
    """Mask `n_masked` positions of every row, a different subset in each."""
    order = torch.rand(samples.shape, generator=generator).argsort(-1)
    return samples.scatter(1, order[:, :n_masked], MASK)


def score(
    problem: Problem,
    arm: Arm,
    completions: Int[torch.Tensor, "n_particles n_completions seq_len"],
) -> Float[torch.Tensor, " n_particles"]:
    """Reduce each particle's completions to the score its arm's potential uses.

    The reduction is `arm.estimator`, exactly the callable that `fk_steer` is handed,
    so the offline comparison measures what the steering runs go on to use.
    """
    n_particles, n_completions, seq_len = completions.shape
    reward_fn = guidance_reward(arm, problem.ehrlich) or true_reward(problem.ehrlich)
    rewards = reward_fn(completions.reshape(-1, seq_len))
    return arm.estimator(rewards.view(n_particles, n_completions))


# %% [markdown]
# ### Experiment 1: feasibility of completions


# %%
def feasibility_by_mask_level(
    problem: Problem, generator: torch.Generator
) -> dict[str, list[float]]:
    """Fraction of completions that are feasible, as a function of mask level.

    At each level `n_masked` positions are masked uniformly at random in every row, and
    each row is completed once by each sampler. Both samplers see the same `x_t`, so
    the comparison is paired and the gap between the curves is attributable to the
    factorisation alone.

    Returns:
        Feasible fractions under each sampler, each of length `seq_len + 1` and indexed
        by the number of masked positions. Index 0 is the unmasked case, feasible by
        construction of `problem.samples`.
    """
    feasibility = {"joint": [1.0], "factorised": [1.0]}
    for n_masked in range(1, problem.seq_len + 1):
        x_t = mask_positions(problem.samples, n_masked, generator)
        completions = {
            "joint": problem.chain.sample_x0_given_xt_via_joint(x_t, 1, generator),
            "factorised": problem.chain.sample_x0_given_xt_via_marginals(
                x_t, 1, generator
            ),
        }
        for sampler, x_0 in completions.items():
            feasible = problem.chain.check_feasibility(x_0.squeeze(1))
            feasibility[sampler].append(feasible.double().mean().item())
    return feasibility


# %% [markdown]
# ### Experiment 2: agreement of the reward estimators


# %%
def reward_estimate_sweep(
    problem: Problem,
    mask_fractions: tuple[float, ...],
    n_particles: int,
    n_completions: int,
    generator: torch.Generator,
) -> dict:
    """Every arm's intermediate reward estimate, at each mask level.

    All the arms at one level share a single `x_t`, and all the factorised ones share a
    single set of completions, so they differ only in how infeasible completions are
    handled. `reference` is the exact joint estimate everything is compared against;
    the joint *arm* is a second, independent joint draw, so its agreement with the
    reference is the Monte-Carlo noise floor of the estimator itself and any further
    disagreement is attributable to the factorisation.

    Returns:
        `mask_fractions`, and one `reference` tensor and one dict of per-arm estimates
        per mask level.
    """
    particles = problem.samples[:n_particles]
    reference, estimates = [], []
    for mask_fraction in mask_fractions:
        x_t = mask_positions(particles, int(mask_fraction * problem.seq_len), generator)
        joint = problem.chain.sample_x0_given_xt_via_joint
        reference.append(score(problem, ARMS[0], joint(x_t, n_completions, generator)))
        control = score(problem, ARMS[0], joint(x_t, n_completions, generator))
        factorised = problem.chain.sample_x0_given_xt_via_marginals(
            x_t, n_completions, generator
        )
        estimates.append({
            arm.name: control if arm.joint_sampler else score(problem, arm, factorised)
            for arm in ARMS
        })
    return {
        "mask_fractions": list(mask_fractions),
        "reference": reference,
        "estimates": estimates,
    }


# %% [markdown]
# ### Experiment 3: FK steering


# %%
def steering_sweep(problem: Problem, generator: torch.Generator) -> dict:
    """Best reward each arm finds over the tilting sweep, against an unguided baseline.

    The metric is the best true reward among a run's *final particles*, averaged over
    `N_RUNS` independent runs. Every arm is judged by the objective regardless of which
    reward its potentials were built from.
    """
    reward_fn = true_reward(problem.ehrlich)
    baseline = best_of_n(
        problem.chain, reward_fn, problem.seq_len, N_PARTICLES, N_RUNS, generator
    )

    means: dict[str, list[float]] = {}
    errors: dict[str, list[float]] = {}
    for arm in ARMS:
        sampler = (
            problem.chain.sample_x0_given_xt_via_joint
            if arm.joint_sampler
            else problem.chain.sample_x0_given_xt_via_marginals
        )
        best = [
            fk_steer(
                problem.chain,
                reward_fn,
                sampler,
                arm.estimator,
                guidance_reward_fn=guidance_reward(arm, problem.ehrlich),
                seq_len=problem.seq_len,
                n_particles=N_PARTICLES,
                n_runs=N_RUNS,
                n_completions=N_COMPLETIONS,
                tilting_strength=tilting_strength,
                generator=generator,
                resample_every=RESAMPLE_EVERY,
                ess_threshold=ESS_THRESHOLD,
            )
            .terminal_rewards.max(dim=1)
            .values
            for tilting_strength in TILTING_STRENGTHS
        ]
        means[arm.name] = [runs.mean().item() for runs in best]
        errors[arm.name] = [runs.std().item() / N_RUNS**0.5 for runs in best]
    return {
        "tilting_strengths": list(TILTING_STRENGTHS),
        "means": means,
        "errors": errors,
        "baseline": baseline.mean().item(),
        "baseline_error": baseline.std().item() / N_RUNS**0.5,
    }


# %% [markdown]
# ## Experiments

# %%
feasibility = {
    family.name: [
        feasibility_by_mask_level(
            problem,
            torch.Generator(device="cpu").manual_seed(
                FEASIBILITY_BASE + problem.problem_seed
            ),
        )
        for problem in REPLICATES[family.name]
    ]
    for family in FAMILIES
}

# %%
estimate_sweeps = {
    family.name: [
        reward_estimate_sweep(
            problem,
            MASK_FRACTIONS,
            N_PARTICLES_ANALYSIS,
            N_COMPLETIONS_ANALYSIS,
            torch.Generator(device="cpu").manual_seed(
                ESTIMATE_BASE + problem.problem_seed
            ),
        )
        for problem in REPLICATES[family.name]
    ]
    for family in FAMILIES
}

# %%
steering = {
    family.name: [
        steering_sweep(
            problem,
            torch.Generator(device="cpu").manual_seed(
                STEERING_BASE + problem.problem_seed
            ),
        )
        for problem in REPLICATES[family.name]
    ]
    for family in FAMILIES
}

# %%
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {"feasibility": feasibility, "estimates": estimate_sweeps, "steering": steering},
    RESULTS_PATH,
)

# %% [markdown]
# ## Figures

# %%
results = torch.load(RESULTS_PATH, weights_only=True)
feasibility, estimate_sweeps, steering = (
    results[key] for key in ("feasibility", "estimates", "steering")
)


# %%
def style(ax: Axes) -> None:
    """House style: grid behind the data, no top or right spine."""
    ax.grid(True, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def arm_handles(*, analysis: bool = False, baseline: bool = False) -> list[Line2D]:
    """Proxy legend entries, so several panels can share one legend."""
    handles = [
        Line2D(
            [],
            [],
            color=arm.colour,
            linestyle=arm.dash,
            marker="o",
            markersize=3,
            label=legend_label(arm, analysis=analysis),
        )
        for arm in ARMS
    ]
    if baseline:
        handles.append(
            Line2D([], [], color="0.45", linestyle="--", label=BASELINE_LABEL)
        )
    return handles


LEGEND_HEIGHT = 0.65
"""Inches to reserve for a two-row legend at `fontsize=6`, including its padding."""


def figure_rc(nrows: int = 1, ncols: int = 2) -> dict:
    """Bundle settings for a grid of panels, with room for a legend beneath them.

    Constrained layout takes the legend's height out of the axes rather than out of the
    figure, so the figure is given enough extra height to pay for it.
    """
    rc = bundles.neurips2024(nrows=nrows, ncols=ncols)
    width, height = rc["figure.figsize"]
    rc["figure.figsize"] = (width, height + LEGEND_HEIGHT)
    return rc


def legend_below(
    fig: plt.Figure, *, analysis: bool = False, baseline: bool = False
) -> None:
    """Put the estimator legend under the panels, where it crowds none of them."""
    fig.legend(
        handles=arm_handles(analysis=analysis, baseline=baseline),
        loc="outside lower center",
        ncol=3,
        fontsize=6,
    )


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to `FIGURE_DIR`, ready to include in the paper."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")


def rank_correlations(
    sweep: dict, *, drop_non_finite: bool = True
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Each arm's rank agreement with the joint estimate, and how much it could score.

    `drop_non_finite` chooses which of two questions the correlation answers, and only
    the first predicts what steering does.

    Keeping the `-inf` particles measures what FK steering experiences. Such a particle
    is not unrankable -- `-inf` ties it at the bottom -- and the sampler does not treat
    it as missing: `_resampling_logits` gives it a `-inf` logit, so it carries zero
    weight and is discarded at the next resampling step, for a reason that has nothing
    to do with its quality. That elimination is the estimator's most consequential
    output, so a correlation must include it.

    Dropping them instead isolates the ranking error alone -- whether an arm orders the
    particles it *can* score the way the joint estimate does -- and leaves `retained`,
    which is unaffected by this argument, to report what that excluded.

    `nan` marks a level where the correlation does not exist: fewer than two particles
    to rank, or all of them scoring the same. That happens to the floored estimator once
    nothing is feasible, and to the `-inf` estimators once nothing is scoreable at all
    -- which is also the point at which a steering run has nothing to select between and
    skips resampling entirely.
    """
    correlation: dict[str, list[float]] = {arm.name: [] for arm in ARMS}
    retained: dict[str, list[float]] = {arm.name: [] for arm in ARMS}
    for reference, estimates in zip(
        sweep["reference"], sweep["estimates"], strict=True
    ):
        for arm in ARMS:
            keep = torch.isfinite(estimates[arm.name])
            retained[arm.name].append(100 * keep.double().mean().item())
            scored = keep if drop_non_finite else torch.ones_like(keep)
            x, y = reference[scored].numpy(), estimates[arm.name][scored].numpy()
            rankable = len(y) >= 2 and y.min() != y.max() and x.min() != x.max()
            correlation[arm.name].append(
                spearmanr(x, y).statistic if rankable else float("nan")
            )
    return correlation, retained


# %% [markdown]
# ### Panels
#
# Each function draws into an axis it is given, so the same panel can appear in the
# main figure and, at more length, in the appendix.


# %%
def draw_feasibility(
    ax: Axes, runs: dict[str, list[dict[str, list[float]]]], *, aggregate: bool = True
) -> None:
    """Feasible fraction against mask level, both samplers and both families.

    With `aggregate` the line is the mean over the instances of a family and the band is
    one standard error of that mean; without it only the first instance is drawn.
    """
    family_markers = ((None, (0.02, 0.1), 2), ("white", (0.07, 0.1), 3))
    for family, (face, markevery, zorder) in zip(
        FAMILIES, family_markers, strict=True
    ):
        for sampler, colour, sampler_label in (
            ("joint", JOINT, "Joint"),
            ("factorised", FACTORISED_OVER_N, "Factorised"),
        ):
            curves = torch.tensor(
                [run[sampler] for run in runs[family.name]], dtype=torch.float64
            )
            if not aggregate:
                curves = curves[:1]
            n_levels = curves.shape[1] - 1
            percentages = [100 * i / n_levels for i in range(n_levels + 1)]
            mean = 100 * curves.mean(0)
            ax.plot(
                percentages,
                mean.tolist(),
                color=colour,
                marker="s",
                markersize=3.2,
                markerfacecolor=face or colour,
                markeredgecolor=colour,
                markeredgewidth=0.8,
                markevery=markevery,
                zorder=zorder,
                label=f"{sampler_label}, {family.label}",
            )
            if aggregate:
                error = 100 * curves.std(0) / len(curves) ** 0.5
                ax.fill_between(
                    percentages,
                    (mean - error).tolist(),
                    (mean + error).tolist(),
                    color=colour,
                    alpha=0.25,
                    linewidth=0,
                    zorder=1,
                )
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 102)
    ax.set_xlabel(r"Positions masked (\%)")
    ax.set_ylabel(r"Completions feasible (\%)")
    style(ax)


def draw_rank_correlation(
    ax: Axes,
    sweeps: list[dict],
    *,
    drop_non_finite: bool = True,
    aggregate: bool = True,
) -> None:
    """Rank agreement with the joint estimate, against mask level.

    `drop_non_finite` ranks only the particles an arm could score; without it every
    particle is ranked, which is what the resampler sees. See `rank_correlations`.

    With `aggregate` each point is the mean over instances and the bar is one standard
    error of it; without it only the first instance is drawn, and it carries no bar
    because a single instance supplies one number per point.

    Fully masked and fully observed levels are not swept: with everything masked every
    particle is identical, so there is no between-particle signal to recover.
    """
    drawn = sweeps if aggregate else sweeps[:1]
    correlations = [
        rank_correlations(sweep, drop_non_finite=drop_non_finite)[0] for sweep in drawn
    ]
    percentages = [100 * fraction for fraction in drawn[0]["mask_fractions"]]
    for arm in ARMS:
        values = torch.tensor(
            [run[arm.name] for run in correlations], dtype=torch.float64
        )
        ax.errorbar(
            percentages,
            values.mean(0).tolist(),
            yerr=(values.std(0) / len(values) ** 0.5).tolist() if aggregate else None,
            color=arm.colour,
            linestyle=arm.dash,
            marker="o",
            markersize=3,
            capsize=1.5,
            elinewidth=0.6,
            label=legend_label(arm, analysis=True),
        )
    ax.axhline(0.0, color="0.6", linewidth=0.6)
    lowest = min(
        (
            rho
            for run in correlations
            for curve in run.values()
            for rho in curve
            if not math.isnan(rho)
        ),
        default=0.0,
    )
    ax.set_ylim(min(-0.25, lowest - 0.05, ax.get_ylim()[0]), 1.05)
    ax.set_xlabel(r"Positions masked (\%)")
    scope = "finite reward only" if drop_non_finite else "all particles"
    ax.set_ylabel(f"Spearman $\\rho$ vs joint\n({scope})", multialignment="center")
    style(ax)


def draw_retention(ax: Axes, sweeps: list[dict], *, aggregate: bool = True) -> None:
    """Fraction of particles left with a finite reward estimate."""
    drawn = sweeps if aggregate else sweeps[:1]
    retentions = [rank_correlations(sweep)[1] for sweep in drawn]
    percentages = [100 * fraction for fraction in drawn[0]["mask_fractions"]]
    for arm in ARMS:
        values = torch.tensor(
            [run[arm.name] for run in retentions], dtype=torch.float64
        )
        ax.errorbar(
            percentages,
            values.mean(0).tolist(),
            yerr=(values.std(0) / len(values) ** 0.5).tolist() if aggregate else None,
            color=arm.colour,
            linestyle=arm.dash,
            marker="o",
            markersize=3,
            capsize=1.5,
            elinewidth=0.6,
        )
    ax.set_ylim(-2, 102)
    ax.set_xlabel(r"Positions masked (\%)")
    ax.set_ylabel("Particles with\nfinite reward (\\%)", multialignment="center")
    style(ax)


def draw_steering(ax: Axes, sweeps: list[dict], *, aggregate: bool = True) -> None:
    """Best reward found against tilting strength, one line per arm.

    With `aggregate` each point is the mean over instances of that instance's mean over
    `N_RUNS` runs, and the bar is one standard error across instances. Without it only
    the first instance is drawn, with its own across-run standard error.
    """
    for arm in ARMS:
        if aggregate:
            values = torch.tensor(
                [run["means"][arm.name] for run in sweeps], dtype=torch.float64
            )
            mean = values.mean(0).tolist()
            error = (values.std(0) / len(values) ** 0.5).tolist()
        else:
            mean, error = sweeps[0]["means"][arm.name], sweeps[0]["errors"][arm.name]
        ax.errorbar(
            sweeps[0]["tilting_strengths"],
            mean,
            yerr=error,
            color=arm.colour,
            linestyle=arm.dash,
            marker="o",
            markersize=3,
            capsize=1.5,
            elinewidth=0.6,
            label=arm.label,
        )
    if aggregate:
        baselines = torch.tensor(
            [run["baseline"] for run in sweeps], dtype=torch.float64
        )
        baseline = baselines.mean().item()
        baseline_error = (baselines.std() / len(baselines) ** 0.5).item()
        ax.axhspan(
            baseline - baseline_error,
            baseline + baseline_error,
            color="0.45",
            alpha=0.15,
            linewidth=0,
        )
    else:
        baseline = sweeps[0]["baseline"]
    ax.axhline(
        baseline, color="0.45", linestyle="--", linewidth=0.8, label=BASELINE_LABEL
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Tilting strength $\lambda$")
    ax.set_ylabel("Best reward among\nfinal particles", multialignment="center")
    style(ax)


# %% [markdown]
# ### Main figure

# %%
with plt.rc_context(figure_rc()):
    fig, (left, right) = plt.subplots(1, 2)
    draw_feasibility(left, feasibility)
    draw_steering(right, steering["long"])
    # (a) is keyed by sampler and family rather than by reward estimator, so it keeps
    # its own legend; (b)'s is shared out to the figure, where there is room for it.
    left.legend(loc="center right", fontsize=5)
    legend_below(fig, baseline=True)
    save(fig, "main")
    plt.show()

# %% [markdown]
# ### Main figure, on a single instance
#
# The same two panels drawn from `SEEDS[0]` alone, with the error bars each experiment
# produces internally: none in (a), where one chain gives one number per mask level, and
# the across-run standard error in (b). Useful for reading a single instance without the
# between-instance spread on top of it; the paper uses the aggregate above.
#

# %%
with plt.rc_context(figure_rc()):
    fig, (left, right) = plt.subplots(1, 2)
    draw_feasibility(left, feasibility, aggregate=False)
    draw_steering(right, steering["long"], aggregate=False)
    left.legend(loc="center right", fontsize=5)
    legend_below(fig, baseline=True)
    save(fig, "main_single_instance")
    plt.show()

# %% [markdown]
# ### Appendix: rank agreement, with retention
#
# One column per problem. The top row is the number that carries over to steering: every
# particle ranked, including those scored `-inf`, which the resampler discards outright
# rather than treats as unmeasured. The two rows below decompose it -- how well an arm
# ranks the particles it can score, and how many of them that is.

# %%
with plt.rc_context(figure_rc(nrows=3, ncols=2)):
    fig, axes = plt.subplots(3, 2, sharex="col", sharey="row")
    for col, family in enumerate(FAMILIES):
        sweeps = estimate_sweeps[family.name]
        draw_rank_correlation(axes[0, col], sweeps, drop_non_finite=False)
        draw_rank_correlation(axes[1, col], sweeps)
        draw_retention(axes[2, col], sweeps)
        axes[0, col].set_title(family.label, fontsize=8)
    # Each column shares an x-axis and each row a scale, so the inner labels would only
    # repeat what the edge ones already say.
    for ax in axes[:2].flat:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    legend_below(fig, analysis=True)
    save(fig, "rank_correlation")
    plt.show()

# %% [markdown]
# ### Appendix: FK steering on both problems
#
# Panel (b) of the main figure, with Ehr-Short alongside it.

# %%
with plt.rc_context(figure_rc()):
    fig, axes = plt.subplots(1, 2, sharex=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        draw_steering(ax, steering[family.name])
        ax.set_title(family.label, fontsize=8)
    legend_below(fig, baseline=True)
    save(fig, "steering")
    plt.show()
