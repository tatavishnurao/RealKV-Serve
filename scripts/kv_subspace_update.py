"""Online low-rank subspace update via incremental PCA with EMA covariance.

The fixed subspace basis (built by build_kv_subspace.py) uses a one-shot SVD
from a 64-token trace. This captures the static KV distribution but cannot
adapt when the prompt, context length, or generation direction moves the KV
statistics elsewhere — leading to "subspace drift" and growing reconstruction
error.

This module provides a batched incremental PCA updater that maintains running
EMA covariance matrices for all layers simultaneously and periodically refreshes
the orthonormal bases via batched eigendecomposition on CPU (fast for small
head_dim×head_dim matrices). No full SVD per step.

Constraints:
- Lightweight (no full SVD per step)
- Operates only on small matrices (head_dim × head_dim covariance)
- Initializable from an offline SVD basis (continuity with existing system)
- Batched eigendecomposition across all layers for efficiency
"""

from __future__ import annotations

import torch


class BatchedIPCAUpdater:
    """Batched incremental PCA across all layers and K/V together.

    Maintains covariance matrices for all layers (K and V) and periodically
    refreshes bases via eigendecomposition. Uses an adaptive trigger: eigh
    only runs when the covariance has changed enough or when the maximum
    interval is reached.
    """

    def __init__(
        self,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        alpha: float = 0.95,
        update_interval: int = 4,
        max_interval: int = 16,
        drift_threshold: float = 0.005,
    ):
        """Create batched IPCA updater for all layers.

        Args:
            w_k: [num_layers, head_dim, rank] K bases from offline SVD.
            w_v: [num_layers, head_dim, rank] V bases from offline SVD.
            alpha: EMA decay factor for covariance.
            update_interval: minimum interval before checking for basis refresh.
            max_interval: force refresh if exceeded (safety bound).
            drift_threshold: Frobenius change in C before triggering eigh.
        """
        n_layers, head_dim, rank = w_k.shape
        self.num_layers = n_layers
        self.head_dim = head_dim
        self.rank = rank
        self.alpha = alpha
        self.update_interval = update_interval
        self.max_interval = max_interval
        self.drift_threshold = drift_threshold
        self.step_count = 0
        self.steps_since_refresh = 0

        self.W_k = w_k.detach().clone()
        self.W_v = w_v.detach().clone()

        device = w_k.device
        num_updaters = 2 * n_layers
        self.C = torch.zeros(num_updaters, head_dim, head_dim, dtype=torch.float32, device=device)
        self.C_prev = torch.zeros(
            num_updaters, head_dim, head_dim, dtype=torch.float32, device=device
        )
        for i in range(n_layers):
            wk = self.W_k[i].to(torch.float32)
            wv = self.W_v[i].to(torch.float32)
            self.C[2 * i] = wk @ wk.t()
            self.C[2 * i + 1] = wv @ wv.t()

    def update_batch(
        self,
        k_vecs: torch.Tensor,
        v_vecs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Update all covariance matrices with new KV vectors.

        Args:
            k_vecs: [num_layers, kv_heads, head_dim] K vectors from last token.
            v_vecs: [num_layers, kv_heads, head_dim] V vectors from last token.

        Returns:
            (W_k, W_v, total_k_drift, total_v_drift) with updated bases and
            drift measured as Frobenius change per layer.
        """
        device = self.C.device
        kf = k_vecs.detach().to(dtype=torch.float32, device=device)
        vf = v_vecs.detach().to(dtype=torch.float32, device=device)

        n_layers = kf.shape[0]
        n_samples = kf.shape[1]

        k_cov = torch.bmm(kf.transpose(1, 2), kf) / n_samples
        v_cov = torch.bmm(vf.transpose(1, 2), vf) / n_samples

        self.C[0::2] = self.alpha * self.C[0::2] + (1.0 - self.alpha) * k_cov
        self.C[1::2] = self.alpha * self.C[1::2] + (1.0 - self.alpha) * v_cov

        self.step_count += 1
        self.steps_since_refresh += 1
        total_k_drift = 0.0
        total_v_drift = 0.0

        should_refresh = self.steps_since_refresh >= self.max_interval

        if not should_refresh and self.steps_since_refresh >= self.update_interval:
            c_diff = self.C - self.C_prev
            frob_norm = float(torch.norm(c_diff))
            if frob_norm / max(1.0, float(torch.norm(self.C_prev))) >= self.drift_threshold:
                should_refresh = True

        if should_refresh:
            C_cpu = self.C.to(device="cpu")
            eigvals, eigvecs = torch.linalg.eigh(C_cpu)
            idx = torch.argsort(eigvals, dim=1, descending=True)
            r = min(self.rank, self.head_dim)
            idx_r = idx[:, :r]

            for i in range(n_layers):
                old_wk = self.W_k[i].detach().clone()
                old_wv = self.W_v[i].detach().clone()

                top_k = eigvecs[2 * i, :, idx_r[2 * i]]
                top_v = eigvecs[2 * i + 1, :, idx_r[2 * i + 1]]

                self.W_k[i] = top_k.to(device=self.W_k.device, dtype=self.W_k.dtype)
                self.W_v[i] = top_v.to(device=self.W_v.device, dtype=self.W_v.dtype)

                total_k_drift += float(
                    torch.norm(
                        self.W_k[i].to(torch.float32) - old_wk.to(torch.float32)
                    )
                )
                total_v_drift += float(
                    torch.norm(
                        self.W_v[i].to(torch.float32) - old_wv.to(torch.float32)
                    )
                )

            self.C_prev = self.C.detach().clone()
            self.steps_since_refresh = 0

        return self.W_k, self.W_v, total_k_drift, total_v_drift

    def compress(
        self,
        past_key_values: object,
        sc,
    ) -> object:
        """Compress cache through current bases.

        Args:
            past_key_values: Current KV cache.
            sc: run_kv_structured_compare module.

        Returns:
            Compressed cache.
        """
        layers = sc.to_legacy_kv(past_key_values)
        rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, (k, v) in enumerate(layers):
            wk = self.W_k[i].to(device=k.device, dtype=k.dtype)
            wv = self.W_v[i].to(device=v.device, dtype=v.dtype)
            k_comp = (k @ wk) @ wk.t()
            v_comp = (v @ wv) @ wv.t()
            rebuilt.append((k_comp.contiguous(), v_comp.contiguous()))
        return sc.from_legacy_kv(rebuilt)

    def update_and_compress(
        self,
        past_key_values: object,
        sc,
    ) -> tuple[object, float, float]:
        """Update with last-token KV and compress full cache.

        Args:
            past_key_values: Current KV cache.
            sc: run_kv_structured_compare module.

        Returns:
            (compressed_cache, total_k_drift, total_v_drift)
        """
        layers = sc.to_legacy_kv(past_key_values)

        k_vecs_list: list[torch.Tensor] = []
        v_vecs_list: list[torch.Tensor] = []
        for i, (k, v) in enumerate(layers):
            k_last = k[:, :, -1:, :].squeeze(2)
            v_last = v[:, :, -1:, :].squeeze(2)
            k_vecs_list.append(k_last.reshape(-1, k_last.shape[-1]))
            v_vecs_list.append(v_last.reshape(-1, v_last.shape[-1]))

        k_vecs = torch.stack(k_vecs_list, dim=0)
        v_vecs = torch.stack(v_vecs_list, dim=0)

        _, _, k_drift, v_drift = self.update_batch(k_vecs, v_vecs)

        rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, (k, v) in enumerate(layers):
            wk = self.W_k[i].to(device=k.device, dtype=k.dtype)
            wv = self.W_v[i].to(device=v.device, dtype=v.dtype)
            k_comp = (k @ wk) @ wk.t()
            v_comp = (v @ wv) @ wv.t()
            rebuilt.append((k_comp.contiguous(), v_comp.contiguous()))

        return sc.from_legacy_kv(rebuilt), k_drift, v_drift

    def get_current_bases(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.W_k, self.W_v
