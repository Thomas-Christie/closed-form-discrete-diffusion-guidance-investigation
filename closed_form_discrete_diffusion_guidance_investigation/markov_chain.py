"""Closed-form sampling of useful quantities from a Markov chain."""

import torch
from jaxtyping import Bool, Float, Int

MASK = -1


class MarkovChain:
    """Markov Chain sampler implementation, with functionality for denoising.

    Provides functionality to:
    - Compute the log probability of an observed sample from the chain.
    - Draw unconditional samples from the chain via ancestral sampling.
    - Draw samples from the chain conditioned on a subset of locations having been
      observed. One method draws such samples by sampling from the unobserved locations
      according to their *marginals*, and one draws such samples by sampling from the
      observed locations according to their *joint* distribution. These use the
      forward-backward algorithm and forward-filtering-backward-sampling respectively.

    Args:
        initial_dist: Initial distribution over states.
        transition_matrix: Matrix A with element A[i, j] representing the
          probability of transitioning from state i to state j.
    """

    def __init__(
        self,
        initial_dist: Float[torch.Tensor, " K"],
        transition_matrix: Float[torch.Tensor, "K K"],
    ) -> None:
        self.n_states = initial_dist.shape[0]
        self.initial_dist = initial_dist.to(torch.float64)
        self.log_initial_dist = torch.log(self.initial_dist)
        self.transition_matrix = transition_matrix.to(torch.float64)
        self.log_transition_matrix = torch.log(self.transition_matrix)

    def check_feasibility(
        self, seqs: Int[torch.Tensor, "N L"]
    ) -> Bool[torch.Tensor, " N"]:
        """Return whether each sequence lies on the chain's support.

        A sequence is feasible exactly when its first token has non-zero initial
        probability and every adjacent pair of tokens has non-zero transition
        probability.

        Args:
            seqs: Sequences to check.

        Returns:
            Boolean mask, true for the feasible sequences.
        """
        starts_feasible = self.initial_dist[seqs[:, 0]] > 0
        bigrams_feasible = (self.transition_matrix[seqs[:, :-1], seqs[:, 1:]] > 0).all(
            -1
        )
        return starts_feasible & bigrams_feasible

    def log_prob(self, seqs: Int[torch.Tensor, "N L"]) -> Float[torch.Tensor, " N"]:
        """Return log probability of supplied sequences.

        Args:
            seqs: Observed sequences.

        Returns:
            Log probability of observed sequences.
        """
        return self.log_initial_dist[seqs[:, 0]] + self.log_transition_matrix[
            seqs[:, :-1], seqs[:, 1:]
        ].sum(-1)

    def sample_full(
        self, seq_len: int, n_seq: int, generator: torch.Generator
    ) -> Int[torch.Tensor, "{n_seq} {seq_len}"]:
        """Samples full sequences via ancestral sampling.

        Args:
            seq_len: Length of sequence to sample.
            n_seq: Number of sequences to samples.
            generator: Torch generator for seeding random operations.

        Returns: n_seq samples from the Markov chain of length seq_len.
        """
        samples = torch.empty(
            (n_seq, seq_len), dtype=torch.int64, device=self.transition_matrix.device
        )
        prev_token_samples = torch.multinomial(
            self.initial_dist, n_seq, replacement=True, generator=generator
        )  # [n_seq]
        samples[:, 0] = prev_token_samples
        for i in range(1, seq_len):
            next_token_probs = self.transition_matrix[prev_token_samples]  # [n_seq, k]
            prev_token_samples = torch.multinomial(
                next_token_probs, 1, replacement=True, generator=generator
            ).squeeze(-1)  # [n_seq]
            samples[:, i] = prev_token_samples
        return samples

    def sample_x0_given_xt_via_marginals(
        self, x_t: Int[torch.Tensor, "N L"], n_samples: int, generator: torch.Generator
    ) -> Int[torch.Tensor, "N {n_samples} L"]:
        """Given partially observed Markov chains, sample the full chains.

        Does so by sampling the missing observations via their *marginal*
        probabilities, rather than the joint, using the forward-backward algorithm.
        This is similar to the typical discrete denoising diffusion probabilistic model
        setup, which assumes that the joint factorises over individual elements. Note
        that the token -1 is treated as a special "mask" token.

        NOTE: This assumes that each row of x_t is a masked observation of a *valid*
        chain i.e. that some full sequence with non-zero probability is consistent with
        the observed tokens.

        Args:
            x_t: Partially observed chains, with masked positions set to ``MASK``.
            n_samples: Number of completions to draw *per row* of ``x_t``.
            generator: Torch generator, for reproducibility.

        Returns:
            Completed sequences. Observed positions are reproduced exactly, but
            masked positions are drawn *independently* of one another, so the
            returned sequences are not guaranteed to have non-zero probability
            under the chain -- this is precisely the factorisation error made by
            an x_0-parametrised denoising diffusion model.
        """
        n_seqs, seq_len = x_t.shape
        device = self.transition_matrix.device
        is_observed = (x_t != MASK).unsqueeze(-1)  # [N, L, 1]
        basis = (x_t.unsqueeze(-1) == torch.arange(self.n_states, device=device)).to(
            torch.float64
        )  # [N, L, n_states]

        alphas = torch.zeros(
            (n_seqs, seq_len, self.n_states), dtype=torch.float64, device=device
        )
        betas = torch.zeros_like(alphas)

        # Forwards
        for i in range(seq_len):
            if i == 0:
                propagated = self.initial_dist.expand(n_seqs, self.n_states)
            else:
                propagated = alphas[:, i - 1] @ self.transition_matrix
            alphas[:, i] = torch.where(is_observed[:, i], basis[:, i], propagated)

        # Backwards
        for i in range(seq_len - 1, -1, -1):
            if i == seq_len - 1:
                propagated = torch.ones(
                    (n_seqs, self.n_states), dtype=torch.float64, device=device
                )
            else:
                propagated = betas[:, i + 1] @ self.transition_matrix.T
            betas[:, i] = torch.where(is_observed[:, i], basis[:, i], propagated)

        unnormalised_probs = alphas * betas
        normalised_probs = unnormalised_probs / torch.sum(
            unnormalised_probs, dim=-1, keepdim=True
        )  # [N, L, n_states]

        samples = torch.multinomial(
            normalised_probs.reshape(n_seqs * seq_len, self.n_states),
            num_samples=n_samples,
            replacement=True,
            generator=generator,
        )  # [N * L, n_samples]
        return samples.view(n_seqs, seq_len, n_samples).transpose(1, 2).contiguous()

    def sample_x0_given_xt_via_joint(
        self, x_t: Int[torch.Tensor, "N L"], n_samples: int, generator: torch.Generator
    ) -> Int[torch.Tensor, "N {n_samples} L"]:
        """Given partially observed Markov chains, sample the full chains.

        Does so by sampling the missing observations via their *joint*
        probabilities, rather than their marginals, using
        forward-filtering-backward-sampling. Note that the token -1 is treated as a
        special "mask" token.

        NOTE: This assumes that each row of x_t is a masked observation of a *valid*
        chain i.e. that some full sequence with non-zero probability is consistent with
        the observed tokens.

        Args:
            x_t: Partially observed chains, with masked positions set to ``MASK``.
            n_samples: Number of completions to draw *per row* of ``x_t``.
            generator: Torch generator, for reproducibility.

        Returns:
            Completed sequences, drawn exactly from the joint posterior over the
            masked positions. Observed positions are reproduced exactly, and every
            returned sequence is guaranteed to have non-zero probability under the
            chain.
        """
        n_seqs, seq_len = x_t.shape
        device = self.transition_matrix.device
        is_observed = (x_t != MASK).unsqueeze(-1)  # [N, L, 1]
        basis = (x_t.unsqueeze(-1) == torch.arange(self.n_states, device=device)).to(
            torch.float64
        )  # [N, L, n_states]

        alphas = torch.zeros(
            (n_seqs, seq_len, self.n_states), dtype=torch.float64, device=device
        )
        samples = torch.zeros(
            (n_seqs, n_samples, seq_len), dtype=torch.long, device=device
        )

        # Forwards
        for i in range(seq_len):
            if i == 0:
                propagated = self.initial_dist.expand(n_seqs, self.n_states)
            else:
                propagated = alphas[:, i - 1] @ self.transition_matrix
            alphas[:, i] = torch.where(is_observed[:, i], basis[:, i], propagated)

        # Sample backwards
        for i in range(seq_len - 1, -1, -1):
            if i == seq_len - 1:
                final_token_probs = alphas[:, -1] / torch.sum(
                    alphas[:, -1], dim=-1, keepdim=True
                )  # [N, n_states]
                drawn = torch.multinomial(
                    final_token_probs,
                    num_samples=n_samples,
                    replacement=True,
                    generator=generator,
                )  # [N, n_samples]
            else:
                next_token_samples = samples[:, :, i + 1]  # [N, n_samples]
                alpha_i = alphas[:, i].unsqueeze(1)  # [N, 1, n_states]
                # The next token's column of A, per (row, sample).
                beta_i = self.transition_matrix.T[
                    next_token_samples
                ]  # [N, n_samples, n_states]
                unnormalised_probs = alpha_i * beta_i
                normalised_probs = unnormalised_probs / torch.sum(
                    unnormalised_probs, dim=-1, keepdim=True
                )  # [N, n_samples, n_states]
                drawn = torch.multinomial(
                    normalised_probs.reshape(n_seqs * n_samples, self.n_states),
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).view(n_seqs, n_samples)  # [N, n_samples]
            samples[:, :, i] = torch.where(is_observed[:, i], x_t[:, i : i + 1], drawn)

        return samples
