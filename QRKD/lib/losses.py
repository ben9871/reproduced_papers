"""KD and RKD (distance & angle) classical losses for MNIST reproduction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    t: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Standard KD with temperature T and mixing alpha (Hinton et al.)."""
    p_s = f.log_softmax(student_logits / t, dim=1)
    p_t = f.softmax(teacher_logits / t, dim=1)
    loss_kd = f.kl_div(p_s, p_t, reduction="batchmean") * (t * t)
    return alpha * loss_kd


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances (full NxN matrix, zeros on diagonal)."""
    x2 = (x * x).sum(dim=1, keepdim=True)  # (N,1)
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    dist2 = dist2.clamp_min(0.0)
    return dist2.clamp_min(1e-12).sqrt()


def rkd_distance_loss(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor
) -> torch.Tensor:
    """Relational KD distance term (matches official RKD: mean over positive distances)."""
    with torch.no_grad():
        td = pairwise_distances(teacher_feat)
        mean_td = td[td > 0].mean()
        td = td / (mean_td + 1e-12)
    sd = pairwise_distances(student_feat)
    mean_sd = sd[sd > 0].mean()
    sd = sd / (mean_sd + 1e-12)
    return f.smooth_l1_loss(sd, td)


def rkd_angle_loss(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor
) -> torch.Tensor:
    """Relational KD angle term (full pairwise angular relations, as in RKD repo)."""
    with torch.no_grad():
        td = teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1)  # (N,N,D)
        norm_td = f.normalize(td, p=2, dim=2)  # (N,N,D)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)  # (N*N*N)
    sd = student_feat.unsqueeze(0) - student_feat.unsqueeze(1)
    norm_sd = f.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
    return f.smooth_l1_loss(s_angle, t_angle)


def simple_fidelity_kernel_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute fidelity-style kernel; backend 'simple' is |<psi_i|psi_j>|^2 with normalized vectors."""
    psi = f.normalize(feat, p=2, dim=1)
    gram = psi @ psi.t()
    return gram.pow(2)


def simple_fidelity_kernel_loss(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor
) -> torch.Tensor:
    """MSE between teacher/student fidelity kernel matrices."""
    with torch.no_grad():
        kt = simple_fidelity_kernel_matrix(teacher_feat)
    ks = simple_fidelity_kernel_matrix(student_feat)
    return f.smooth_l1_loss(ks, kt)


class DistillationLoss(nn.Module):
    """Container for KD + RKD terms (distance/angle) and fidelity-kernel QRKD."""

    def __init__(
        self,
        kd: float = 0.5,
        dr: float = 0.1,
        ar: float = 0.1,
        qk: float = 0.0,
        qk_backend: str = "simple",
        qk_n_modes: int | None = None,
        qk_n_photons: int | None = None,
        temperature: float = 4.0,
        kd_alpha: float = 0.5,
    ):
        super().__init__()
        self.kd = kd
        self.dr = dr
        self.ar = ar
        self.qk = qk
        self.qk_backend = qk_backend
        self.qk_n_modes = qk_n_modes
        self.qk_n_photons = qk_n_photons
        self.temperature = temperature
        self.kd_alpha = kd_alpha
        self._merlin_kernel = None
        self._qiskit_kernel = None
        self._qiskit_dim: int | None = None

    def forward(self, logits_s, logits_t, feat_s, feat_t):
        loss = torch.zeros((), device=logits_s.device)
        if self.kd:
            loss = loss + self.kd * kd_loss(
                logits_s, logits_t, t=self.temperature, alpha=self.kd_alpha
            )
        if self.dr:
            loss = loss + self.dr * rkd_distance_loss(feat_s, feat_t)
        if self.ar:
            loss = loss + self.ar * rkd_angle_loss(feat_s, feat_t)
        if self.qk:
            # Ensure student/teacher feature vectors share the same dimensionality for kernel backends.
            feat_s_qk = feat_s
            feat_t_qk = feat_t
            if feat_s.shape[1] != feat_t.shape[1]:
                target_dim = min(feat_s.shape[1], feat_t.shape[1])
                feat_s_qk = feat_s[:, :target_dim]
                feat_t_qk = feat_t[:, :target_dim]
            if self.qk_backend.lower() == "merlin":
                # Build once per feature dimensionality; merlin kernels run on CPU
                if (
                    self._merlin_kernel is None
                    or getattr(self._merlin_kernel, "input_size", None)
                    != feat_s_qk.shape[1]
                ):
                    from merlin.algorithms.kernels import FidelityKernel  # type: ignore

                    n_modes = self.qk_n_modes or max(2, min(feat_s_qk.shape[1], 4))
                    n_photons = self.qk_n_photons or min(feat_s_qk.shape[1], n_modes)
                    self._merlin_kernel = FidelityKernel.simple(
                        input_size=feat_s_qk.shape[1],
                        n_modes=n_modes,
                        n_photons=n_photons,
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                        force_psd=True,
                        shots=0,
                    )
                ks = self._merlin_kernel(
                    feat_s_qk.detach().to("cpu", dtype=torch.float32)
                ).to(device=feat_s.device, dtype=feat_s.dtype)
                with torch.no_grad():
                    kt = self._merlin_kernel(
                        feat_t_qk.detach().to("cpu", dtype=torch.float32)
                    ).to(device=feat_s.device, dtype=feat_s.dtype)
                loss = loss + self.qk * f.smooth_l1_loss(ks, kt)
            elif self.qk_backend.lower() == "qiskit":
                if (
                    self._qiskit_kernel is None
                    or self._qiskit_dim != feat_s_qk.shape[1]
                ):
                    from qiskit.circuit.library import ZZFeatureMap  # type: ignore
                    from qiskit_machine_learning.kernels import (
                        FidelityQuantumKernel,  # type: ignore
                    )

                    self._qiskit_dim = int(feat_s_qk.shape[1])
                    fmap = ZZFeatureMap(feature_dimension=self._qiskit_dim)
                    self._qiskit_kernel = FidelityQuantumKernel(feature_map=fmap)
                xs = feat_s_qk.detach().to("cpu", dtype=torch.float64).numpy()
                xt = feat_t_qk.detach().to("cpu", dtype=torch.float64).numpy()
                ks_np = self._qiskit_kernel.evaluate(xs, xs)  # type: ignore[union-attr]
                kt_np = self._qiskit_kernel.evaluate(xt, xt)  # type: ignore[union-attr]
                ks = torch.as_tensor(ks_np, dtype=feat_s.dtype, device=feat_s.device)
                kt = torch.as_tensor(kt_np, dtype=feat_s.dtype, device=feat_s.device)
                loss = loss + self.qk * f.smooth_l1_loss(ks, kt)
            else:
                loss = loss + self.qk * simple_fidelity_kernel_loss(
                    feat_s_qk, feat_t_qk
                )
        return loss
