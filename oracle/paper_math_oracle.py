"""Paper-math oracle for the Multiscreen architecture.

This module is intentionally small, explicit, and slow-ish.  It is meant for
correctness tests on tiny tensors, not for pretraining or long-context speed
measurements.

The implementation follows the equations in the Multiscreen paper:

  - row-normalized tied input/output embeddings with learned s_E and s_F
  - per-layer residual sum of parallel gated screening tiles
  - q/k/v unit normalization
  - MiPE on the first two q/k coordinates
  - Trim relevance with paper acceptance width r = sigmoid(s_r)
  - causal distance Softmask
  - value aggregation followed by TanhNorm
  - tanh(SiLU(.)) gate and per-head residual scale exp(s_O)

A helper is included for copying weights from the current Hugging Face port.  The
HF port supplied in this project parameterizes Trim with an inverse acceptance
width, `inv_r = exp(sr) + 1`, while the paper parameterizes acceptance width as
`r = sigmoid(s_r)`.  These are equivalent if `s_r_paper = -s_r_hf`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, NamedTuple, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


PositionRule = Literal["paper", "hf_mod_after_max_position"]
ComputeDTypeRule = Literal["fp32", "reference"]
ScreeningCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class PaperMultiscreenConfig:
    """Minimal config for the paper-math oracle.

    Defaults mirror the paper-style architecture where possible.  For fast unit
    tests, construct much smaller configs manually or with ``from_psi`` using a
    tiny vocabulary.
    """

    vocab_size: int = 50_257
    hidden_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    key_dim: int = 16
    value_dim: int = 64
    max_position_embeddings: int = 4_096
    mipe_threshold: float = 256.0
    initializer_range: float = 0.1
    norm_eps: float = 1e-12
    tanh_norm_eps: float = 1e-8
    labels_are_shifted: bool = False
    # The paper formula uses absolute position i.  The current HF port also has
    # an optional modulo behavior after max_position_embeddings.  Keep paper as
    # the default; use the HF rule only when intentionally reproducing that port
    # outside the training context.
    position_rule: PositionRule = "paper"
    # Numerical reference mode.  The paper/oracle default is stable fp32
    # auxiliary math for MiPE and Softmask under bf16/fp16.  For compatibility
    # with the original unofficial PyTorch reference implementation, set these
    # to "reference" so the auxiliary MiPE/Softmask math follows the incoming
    # tensor dtype.  This mainly matters in low precision when exercising the
    # HF/reference max-position modulo branch.
    mipe_compute_dtype: ComputeDTypeRule = "fp32"
    softmask_compute_dtype: ComputeDTypeRule = "fp32"
    # The cache API is intentionally narrow for P0: caches must represent a
    # contiguous prefix starting at position 0, and suffix calls must use
    # start_pos == past_len.  Nonzero no-cache start_pos is rejected because the
    # current distance Softmask assumes key positions 0..T_total-1.
    strict_cache_positions: bool = True

    @classmethod
    def from_psi(
        cls,
        psi: int,
        *,
        vocab_size: int = 50_257,
        max_seq_len: int = 4_096,
        **overrides: Any,
    ) -> "PaperMultiscreenConfig":
        """Paper-style scaling: N_L = N_H = Psi and d_E = Psi^2."""

        return cls(
            vocab_size=vocab_size,
            hidden_size=psi * psi,
            num_hidden_layers=psi,
            num_attention_heads=psi,
            max_position_embeddings=max_seq_len,
            **overrides,
        )

    @classmethod
    def from_hf_config(cls, hf_config: Any, **overrides: Any) -> "PaperMultiscreenConfig":
        """Build an oracle config from a MultiscreenConfig-like HF object."""

        data = {
            "vocab_size": int(_get_attr_any(hf_config, "vocab_size")),
            "hidden_size": int(_get_attr_any(hf_config, "hidden_size", "hidden_dim")),
            "num_hidden_layers": int(_get_attr_any(hf_config, "num_hidden_layers", "num_layers")),
            "num_attention_heads": int(_get_attr_any(hf_config, "num_attention_heads", "num_heads")),
            "key_dim": int(_get_attr_any(hf_config, "key_dim")),
            "value_dim": int(_get_attr_any(hf_config, "value_dim")),
            "max_position_embeddings": int(_get_attr_any(hf_config, "max_position_embeddings", "max_seq_len")),
            "mipe_threshold": float(getattr(hf_config, "mipe_threshold", 256.0)),
            "initializer_range": float(getattr(hf_config, "initializer_range", 0.1)),
            "mipe_compute_dtype": str(getattr(hf_config, "mipe_compute_dtype", "fp32")),
            "softmask_compute_dtype": str(getattr(hf_config, "softmask_compute_dtype", "fp32")),
        }
        data.update(overrides)
        return cls(**data)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.key_dim < 2:
            raise ValueError("key_dim must be at least 2 because MiPE rotates q/k coordinates 0 and 1")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if self.mipe_threshold <= 0:
            raise ValueError("mipe_threshold must be positive")
        if self.position_rule not in {"paper", "hf_mod_after_max_position"}:
            raise ValueError(f"unknown position_rule: {self.position_rule!r}")
        if self.mipe_compute_dtype not in {"fp32", "reference"}:
            raise ValueError(f"unknown mipe_compute_dtype: {self.mipe_compute_dtype!r}")
        if self.softmask_compute_dtype not in {"fp32", "reference"}:
            raise ValueError(f"unknown softmask_compute_dtype: {self.softmask_compute_dtype!r}")


class OracleOutput(NamedTuple):
    """Simple output container that avoids any dependency on Transformers."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: Optional[torch.Tensor]
    past_key_values: Optional[tuple[ScreeningCache, ...]]
    all_hidden_states: Optional[tuple[torch.Tensor, ...]]
    aux: Optional[dict[str, list[torch.Tensor]]]


def dtype_safe_eps(x: torch.Tensor, eps: float) -> float:
    """Return an epsilon that remains positive in ``x``'s dtype.

    The paper only requires a small positive stability constant.  In fp16,
    values such as 1e-12 or 1e-8 may underflow to zero, so zero-norm vectors can
    still create NaNs.  For fp32/float64 this returns the configured epsilon; for
    fp16 it raises the floor to ``torch.finfo(dtype).tiny``.
    """

    eps = float(eps)
    if not torch.is_floating_point(x):
        return eps
    return max(eps, float(torch.finfo(x.dtype).tiny))


def select_aux_compute_dtype(dtype: torch.dtype, rule: ComputeDTypeRule) -> torch.dtype:
    """Return auxiliary math dtype for MiPE/Softmask.

    ``rule="fp32"`` keeps low-precision runs numerically stable and is the
    oracle's paper-oriented default.  ``rule="reference"`` follows the input
    dtype and is only intended for bitwise-style comparison with the original
    unofficial PyTorch reference/HF compatibility path.
    """

    if rule == "reference":
        return dtype
    if rule == "fp32":
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    raise ValueError(f"unknown auxiliary compute dtype rule: {rule!r}")


def unit_normalize(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalize along the last axis with the denominator clipped from below."""

    return x / x.norm(dim=-1, keepdim=True).clamp_min(dtype_safe_eps(x, eps))


def paper_acceptance_width(s_r: torch.Tensor) -> torch.Tensor:
    """Paper equation: r = sigmoid(s_r)."""

    return torch.sigmoid(s_r)


def trim_relevance(similarity: torch.Tensor, acceptance_width: torch.Tensor) -> torch.Tensor:
    """Paper Trim transform.

    Args:
        similarity: q-k similarities with shape ``(..., H, T_q, T_k)`` or any
            broadcast-compatible shape whose head axis matches ``acceptance_width``.
        acceptance_width: per-head r values in ``(0, 1)`` with shape ``(H,)``.

    Returns:
        ``max(1 - (1 - s_ij) / r, 0)^2``.
    """

    # The common call path uses similarity shape (B, H, T_q, T_k).
    if similarity.dim() < 3:
        raise ValueError("similarity must include a head axis")
    if acceptance_width.numel() != similarity.shape[1]:
        raise ValueError(
            f"acceptance_width has {acceptance_width.numel()} heads, "
            f"but similarity.shape[1] is {similarity.shape[1]}"
        )
    r = acceptance_width.to(device=similarity.device, dtype=similarity.dtype).view(
        1, similarity.shape[1], *([1] * (similarity.dim() - 2))
    )
    return torch.clamp(1.0 - (1.0 - similarity) / r, min=0.0).square()


def tanh_norm(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """TanhNorm(x) = tanh(||x||) / ||x|| * x, with the limiting value x."""

    safe_eps = dtype_safe_eps(x, eps)
    norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > safe_eps, torch.tanh(norm) / norm.clamp_min(safe_eps), torch.ones_like(norm))
    return scale * x


def mipe_gamma(w: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """Paper gamma(w) used by MiPE."""

    return torch.where(
        w < threshold,
        0.5 * (torch.cos(math.pi * w / threshold) + 1.0),
        torch.zeros_like(w),
    )


def _effective_positions(
    positions: torch.Tensor,
    *,
    w: torch.Tensor,
    max_position_embeddings: int,
    position_rule: PositionRule,
) -> torch.Tensor:
    """Return position values for MiPE.

    ``position_rule='paper'`` is literal equation (6).  ``'hf_mod_after_max_position'``
    reproduces the extra modulo branch in the current HF port for long positions.
    """

    if position_rule == "paper":
        return positions.unsqueeze(-1).expand(-1, w.numel())
    if position_rule == "hf_mod_after_max_position":
        pos = positions.unsqueeze(-1)
        w_b = w.unsqueeze(0)
        return torch.where(pos >= max_position_embeddings, torch.remainder(pos, w_b), pos)
    raise ValueError(f"unknown position_rule: {position_rule!r}")


def apply_mipe(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    start_pos: int,
    threshold: float,
    max_position_embeddings: int,
    position_rule: PositionRule = "paper",
    compute_dtype_rule: ComputeDTypeRule = "fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MiPE to q and k.

    Args:
        q, k: tensors of shape ``(B, T, H, dK)``.
        w: per-head screening windows, shape ``(H,)``.
        start_pos: absolute position of the first q/k token in this call.
    """

    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.shape[-1] < 2:
        raise ValueError("MiPE requires key_dim >= 2")

    bsz, seq_len, num_heads, _ = q.shape
    del bsz, num_heads

    compute_dtype = select_aux_compute_dtype(q.dtype, compute_dtype_rule)
    w_f = w.to(device=q.device, dtype=compute_dtype)
    gamma = mipe_gamma(w_f, threshold=threshold)
    positions = torch.arange(start_pos, start_pos + seq_len, device=q.device, dtype=compute_dtype)
    pos = _effective_positions(
        positions,
        w=w_f,
        max_position_embeddings=max_position_embeddings,
        position_rule=position_rule,
    )
    angles = pos * (math.pi * gamma / w_f).unsqueeze(0)  # (T, H)
    cos_a = torch.cos(angles).to(dtype=q.dtype).unsqueeze(0)  # (1, T, H)
    sin_a = torch.sin(angles).to(dtype=q.dtype).unsqueeze(0)

    q_rot = q.clone()
    k_rot = k.clone()

    q0, q1 = q[..., 0], q[..., 1]
    k0, k1 = k[..., 0], k[..., 1]
    q_rot[..., 0] = q0 * cos_a - q1 * sin_a
    q_rot[..., 1] = q0 * sin_a + q1 * cos_a
    k_rot[..., 0] = k0 * cos_a - k1 * sin_a
    k_rot[..., 1] = k0 * sin_a + k1 * cos_a
    return q_rot, k_rot


def causal_distance_softmask(
    *,
    t_new: int,
    t_total: int,
    start_pos: int,
    w: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    compute_dtype_rule: ComputeDTypeRule = "fp32",
) -> torch.Tensor:
    """Paper Softmask, vectorized over heads.

    Returns a mask of shape ``(1, H, t_new, t_total)``.  It uses the same cached
    decoding convention as the HF port: keys are at absolute positions
    ``0, ..., t_total - 1`` and new queries are at ``start_pos, ..., start_pos+t_new-1``.
    For standard full-context oracle tests, use ``start_pos=0`` and no cache.
    """

    compute_dtype = select_aux_compute_dtype(dtype, compute_dtype_rule)
    w_f = w.to(device=device, dtype=compute_dtype).view(-1, 1, 1)  # (H,1,1)
    q_pos = torch.arange(start_pos, start_pos + t_new, device=device, dtype=compute_dtype)
    k_pos = torch.arange(t_total, device=device, dtype=compute_dtype)
    rel = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # (Tq,Tk), j-i
    valid = (rel <= 0) & (rel > -w_f)
    mask = 0.5 * (torch.cos(math.pi * rel.unsqueeze(0) / w_f) + 1.0)
    mask = mask * valid.to(dtype=mask.dtype)
    return mask.unsqueeze(0).to(dtype=dtype)  # (1,H,Tq,Tk)


class PaperGatedScreeningLayer(nn.Module):
    """One paper Multiscreen residual layer containing H parallel tiles."""

    def __init__(self, config: PaperMultiscreenConfig, *, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        h = config.num_attention_heads
        e = config.hidden_size
        dk = config.key_dim
        dv = config.value_dim
        init = config.initializer_range

        # Per-head matrices use the paper orientation: x_i @ W_Q[h].
        self.W_Q = nn.Parameter(torch.empty(h, e, dk))
        self.W_K = nn.Parameter(torch.empty(h, e, dk))
        self.W_V = nn.Parameter(torch.empty(h, e, dv))
        self.W_G = nn.Parameter(torch.empty(h, e, dv))
        self.W_O = nn.Parameter(torch.empty(h, dv, e))

        self.s_w = nn.Parameter(torch.linspace(0.0, math.log(config.mipe_threshold), h))
        self.s_r = nn.Parameter(torch.zeros(h))  # paper s_r, so r = sigmoid(s_r)
        self.s_O = nn.Parameter(
            torch.full((h,), math.log(1.0 / math.sqrt(config.num_attention_heads * config.num_hidden_layers)))
        )

        nn.init.normal_(self.W_Q, mean=0.0, std=init / math.sqrt(dk))
        nn.init.normal_(self.W_K, mean=0.0, std=init / math.sqrt(dk))
        nn.init.normal_(self.W_V, mean=0.0, std=init / math.sqrt(dv))
        nn.init.normal_(self.W_G, mean=0.0, std=init)
        nn.init.normal_(self.W_O, mean=0.0, std=init / math.sqrt(e))

    def forward(
        self,
        x: torch.Tensor,
        *,
        start_pos: int = 0,
        past_kv: Optional[ScreeningCache] = None,
        use_cache: bool = False,
        key_attention_mask: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Optional[ScreeningCache], Optional[dict[str, torch.Tensor]]]:
        """Apply one residual layer.

        Args:
            x: hidden states, shape ``(B, T, dE)``.
            key_attention_mask: optional shape ``(B, T_total)``.
            query_attention_mask: optional shape ``(B, T)``.  This is not part of
                the paper equations; it is only here to match HF padding behavior
                in tests.
        """

        cfg = self.config
        bsz, seq_len, _ = x.shape
        w = self.s_w.exp() + 1.0
        r = paper_acceptance_width(self.s_r)

        q = torch.einsum("bte,hek->bthk", x, self.W_Q)
        k_new = torch.einsum("bte,hek->bthk", x, self.W_K)
        v_new = torch.einsum("bte,hev->bthv", x, self.W_V)
        g = torch.einsum("bte,hev->bthv", x, self.W_G)

        q = unit_normalize(q, eps=cfg.norm_eps)
        k_new = unit_normalize(k_new, eps=cfg.norm_eps)
        v_new = unit_normalize(v_new, eps=cfg.norm_eps)

        q, k_new = apply_mipe(
            q,
            k_new,
            w,
            start_pos=start_pos,
            threshold=cfg.mipe_threshold,
            max_position_embeddings=cfg.max_position_embeddings,
            position_rule=cfg.position_rule,
            compute_dtype_rule=cfg.mipe_compute_dtype,
        )

        q_h = q.transpose(1, 2)  # (B,H,T,dK)
        k_new_h = k_new.transpose(1, 2)
        v_new_h = v_new.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k_full = torch.cat([past_k.to(device=x.device), k_new_h], dim=2)
            v_full = torch.cat([past_v.to(device=x.device), v_new_h], dim=2)
        else:
            k_full = k_new_h
            v_full = v_new_h

        t_total = int(k_full.shape[2])
        similarity = torch.einsum("bhtk,bhsk->bhts", q_h, k_full)
        alpha = trim_relevance(similarity, r)
        mask = causal_distance_softmask(
            t_new=seq_len,
            t_total=t_total,
            start_pos=start_pos,
            w=w,
            dtype=similarity.dtype,
            device=similarity.device,
            compute_dtype_rule=cfg.softmask_compute_dtype,
        )
        relevance = alpha * mask

        if key_attention_mask is not None:
            if key_attention_mask.shape != (bsz, t_total):
                raise ValueError(
                    f"key_attention_mask must have shape {(bsz, t_total)}, got {tuple(key_attention_mask.shape)}"
                )
            relevance = relevance * key_attention_mask.to(device=x.device, dtype=relevance.dtype).view(bsz, 1, 1, t_total)

        h = torch.einsum("bhts,bhsv->bhtv", relevance, v_full)
        u = tanh_norm(h, eps=cfg.tanh_norm_eps).transpose(1, 2)  # (B,T,H,dV)

        g_hat = torch.tanh(F.silu(g))
        z = u * g_hat
        if query_attention_mask is not None:
            if query_attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"query_attention_mask must have shape {(bsz, seq_len)}, got {tuple(query_attention_mask.shape)}"
                )
            z = z * query_attention_mask.to(device=x.device, dtype=z.dtype).view(bsz, seq_len, 1, 1)

        z = z * self.s_O.exp().view(1, 1, cfg.num_attention_heads, 1)
        delta = torch.einsum("bthv,hve->bte", z, self.W_O)
        if query_attention_mask is not None:
            delta = delta * query_attention_mask.to(device=x.device, dtype=delta.dtype).unsqueeze(-1)

        new_kv = (k_full, v_full) if use_cache else None
        y = x + delta

        aux = None
        if return_aux:
            aux = {
                "w": w.detach(),
                "r": r.detach(),
                "similarity": similarity.detach(),
                "trim": alpha.detach(),
                "softmask": mask.detach(),
                "relevance": relevance.detach(),
                "pre_tanhnorm": h.detach(),
                "screening_output": u.detach(),
                "gate": g_hat.detach(),
                "delta": delta.detach(),
            }
        return y, new_kv, aux

    @torch.no_grad()
    def copy_from_hf_block(self, hf_block: Any, *, hf_uses_inverse_sr: bool = True) -> None:
        """Copy a layer block from the current HF port into paper-oracle layout."""

        h = self.config.num_attention_heads
        e = self.config.hidden_size
        dk = self.config.key_dim
        dv = self.config.value_dim

        self.W_Q.copy_(hf_block.q_proj.weight.detach().view(h, dk, e).permute(0, 2, 1))
        self.W_K.copy_(hf_block.k_proj.weight.detach().view(h, dk, e).permute(0, 2, 1))
        self.W_V.copy_(hf_block.v_proj.weight.detach().view(h, dv, e).permute(0, 2, 1))
        self.W_G.copy_(hf_block.g_proj.weight.detach().view(h, dv, e).permute(0, 2, 1))
        self.W_O.copy_(hf_block.o_proj.weight.detach().transpose(0, 1).contiguous().view(h, dv, e))
        self.s_w.copy_(hf_block.sw.detach())
        self.s_r.copy_((-hf_block.sr if hf_uses_inverse_sr else hf_block.sr).detach())
        self.s_O.copy_(hf_block.sO.detach())

    @torch.no_grad()
    def copy_from_hf_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        layer_idx: int,
        prefix: str = "",
        hf_uses_inverse_sr: bool = True,
    ) -> None:
        """Copy one layer from HF-style state-dict keys."""

        h = self.config.num_attention_heads
        e = self.config.hidden_size
        dk = self.config.key_dim
        dv = self.config.value_dim
        base = f"{prefix}layers.{layer_idx}.block."

        def get(name: str) -> torch.Tensor:
            key = base + name
            if key not in state_dict:
                raise KeyError(f"missing state_dict key: {key}")
            return state_dict[key].detach()

        self.W_Q.copy_(get("q_proj.weight").view(h, dk, e).permute(0, 2, 1))
        self.W_K.copy_(get("k_proj.weight").view(h, dk, e).permute(0, 2, 1))
        self.W_V.copy_(get("v_proj.weight").view(h, dv, e).permute(0, 2, 1))
        self.W_G.copy_(get("g_proj.weight").view(h, dv, e).permute(0, 2, 1))
        self.W_O.copy_(get("o_proj.weight").transpose(0, 1).contiguous().view(h, dv, e))
        self.s_w.copy_(get("sw"))
        self.s_r.copy_((-get("sr") if hf_uses_inverse_sr else get("sr")))
        self.s_O.copy_(get("sO"))


class PaperMultiscreenForCausalLM(nn.Module):
    """Standalone paper-math Multiscreen causal LM oracle."""

    def __init__(self, config: PaperMultiscreenConfig) -> None:
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        self.s_E = nn.Parameter(torch.tensor(0.0))
        self.s_F = nn.Parameter(torch.tensor(math.log(math.sqrt(config.hidden_size))))
        self.layers = nn.ModuleList(
            [PaperGatedScreeningLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        nn.init.normal_(self.W_E, mean=0.0, std=config.initializer_range / math.sqrt(config.hidden_size))

    def normalized_embeddings(self) -> torch.Tensor:
        return unit_normalize(self.W_E, eps=self.config.norm_eps)

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return F.embedding(input_ids, self.normalized_embeddings()) * self.s_E.exp()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.normalized_embeddings().transpose(0, 1)) * self.s_F.exp()

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Sequence[ScreeningCache]] = None,
        use_cache: bool = False,
        start_pos: Optional[int] = None,
        logits_to_keep: int = 0,
        labels_are_shifted: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_aux: bool = False,
    ) -> OracleOutput:
        """Run the causal LM oracle.

        ``attention_mask`` and cache behavior are included only to make tests
        against the HF wrapper possible.  For literal paper full-context tests,
        pass no mask, no cache, and ``start_pos=0``.
        """

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape (B,T), got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape
        x = self.embed_input_ids(input_ids)

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            if len(past_key_values) != len(self.layers):
                raise ValueError(f"past_key_values length {len(past_key_values)} != num layers {len(self.layers)}")
            past_len = int(past_key_values[0][0].shape[2])
        else:
            past_key_values = None
        had_past = past_key_values is not None
        if start_pos is None:
            start_pos = past_len
        else:
            start_pos = int(start_pos)

        if self.config.strict_cache_positions:
            if had_past and start_pos != past_len:
                raise ValueError(
                    "paper_math_oracle supports cached decoding only for contiguous prefixes starting at 0; "
                    f"got start_pos={start_pos} but past_len={past_len}."
                )
            if (not had_past) and start_pos != 0:
                raise ValueError(
                    "paper_math_oracle full-context/no-cache calls require start_pos=0. "
                    "Use function-level apply_mipe tests for absolute-position MiPE checks, or provide a "
                    "contiguous prefix cache with start_pos == past_len."
                )

        key_mask, query_mask = self._prepare_attention_masks(
            attention_mask=attention_mask,
            batch_size=bsz,
            past_length=past_len,
            seq_len=seq_len,
            total_length=past_len + seq_len,
            device=x.device,
        )

        new_kvs: list[ScreeningCache] = []
        all_hidden: Optional[list[torch.Tensor]] = [] if output_hidden_states else None
        aux: Optional[dict[str, list[torch.Tensor]]] = {} if return_aux else None

        for layer_idx, layer in enumerate(self.layers):
            if all_hidden is not None:
                all_hidden.append(x)
            past_layer = past_key_values[layer_idx] if past_key_values is not None else None
            x, new_kv, layer_aux = layer(
                x,
                start_pos=start_pos,
                past_kv=past_layer,
                use_cache=use_cache,
                key_attention_mask=key_mask,
                query_attention_mask=query_mask,
                return_aux=return_aux,
            )
            if use_cache:
                if new_kv is None:
                    raise RuntimeError("internal error: use_cache=True but layer returned no cache")
                new_kvs.append(new_kv)
            if aux is not None and layer_aux is not None:
                for key, value in layer_aux.items():
                    aux.setdefault(key, []).append(value)

        if all_hidden is not None:
            all_hidden.append(x)

        logits_input = x[:, -logits_to_keep:, :] if labels is None and logits_to_keep and logits_to_keep > 0 else x
        logits = self.compute_logits(logits_input)
        loss = self._compute_loss(
            logits=logits,
            labels=labels,
            attention_mask=attention_mask,
            labels_are_shifted=self.config.labels_are_shifted if labels_are_shifted is None else labels_are_shifted,
        )
        return OracleOutput(
            logits=logits,
            hidden_states=x,
            loss=loss,
            past_key_values=tuple(new_kvs) if use_cache else None,
            all_hidden_states=tuple(all_hidden) if all_hidden is not None else None,
            aux=aux,
        )

    def _compute_loss(
        self,
        *,
        logits: torch.Tensor,
        labels: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        labels_are_shifted: bool,
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                "loss requires logits and labels to have the same first two dimensions; "
                f"got logits {tuple(logits.shape)} labels {tuple(labels.shape)}"
            )
        loss_labels = labels.to(device=logits.device).clone()
        loss_mask = None
        if attention_mask is not None:
            loss_mask = self._slice_loss_attention_mask(
                attention_mask=attention_mask,
                target_length=loss_labels.shape[1],
                device=loss_labels.device,
            )
            loss_labels = loss_labels.masked_fill(loss_mask == 0, -100)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if labels_are_shifted:
            return loss_fct(logits.reshape(-1, self.config.vocab_size), loss_labels.reshape(-1))
        if logits.shape[1] < 2:
            return logits.new_zeros(())
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = loss_labels[:, 1:].contiguous()
        if loss_mask is not None:
            valid_shift = (loss_mask[:, :-1] != 0) & (loss_mask[:, 1:] != 0)
            shift_labels = shift_labels.masked_fill(~valid_shift, -100)
        return loss_fct(shift_logits.reshape(-1, self.config.vocab_size), shift_labels.reshape(-1))

    @staticmethod
    def _slice_loss_attention_mask(
        *,
        attention_mask: torch.Tensor,
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=device)
        if mask.dim() != 2:
            raise ValueError("attention_mask must have shape (B,T)")
        if mask.shape[1] == target_length:
            return mask
        if mask.shape[1] > target_length:
            return mask[:, -target_length:]
        prefix = torch.ones(mask.shape[0], target_length - mask.shape[1], device=device, dtype=mask.dtype)
        return torch.cat([prefix, mask], dim=1)

    @staticmethod
    def _prepare_attention_masks(
        *,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        past_length: int,
        seq_len: int,
        total_length: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            return None, None
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape (B,T)")
        if attention_mask.shape[0] != batch_size:
            raise ValueError(f"attention_mask batch size {attention_mask.shape[0]} != {batch_size}")
        mask = attention_mask.to(device=device)
        mask_len = int(mask.shape[1])
        if mask_len == total_length:
            return mask, mask[:, -seq_len:]
        if mask_len == seq_len:
            if past_length > 0:
                prefix = torch.ones(batch_size, past_length, device=device, dtype=mask.dtype)
                return torch.cat([prefix, mask], dim=1), mask
            return mask, mask
        if mask_len > total_length:
            key_mask = mask[:, -total_length:]
            return key_mask, key_mask[:, -seq_len:]
        # mask_len < total_length: assume missing older cached positions are valid.
        prefix = torch.ones(batch_size, total_length - mask_len, device=device, dtype=mask.dtype)
        key_mask = torch.cat([prefix, mask], dim=1)
        return key_mask, key_mask[:, -seq_len:]

    @torch.no_grad()
    def copy_from_hf_model(self, hf_model: Any, *, hf_uses_inverse_sr: bool = True) -> None:
        """Copy weights from an instantiated HF Multiscreen model.

        Accepts either ``MultiscreenForCausalLM`` or the bare ``MultiscreenModel``.
        Set ``hf_uses_inverse_sr=False`` only if the HF port has already been
        changed to use paper ``r = sigmoid(s_r)`` directly.
        """

        root = getattr(hf_model, "multiscreen", hf_model)
        self.W_E.copy_(root.embed.weight.detach())
        self.s_E.copy_(root.s_E.detach())
        self.s_F.copy_(root.s_F.detach())
        if len(root.layers) != len(self.layers):
            raise ValueError(f"HF model has {len(root.layers)} layers, oracle has {len(self.layers)}")
        for oracle_layer, hf_layer in zip(self.layers, root.layers):
            hf_block = getattr(hf_layer, "block", hf_layer)
            oracle_layer.copy_from_hf_block(hf_block, hf_uses_inverse_sr=hf_uses_inverse_sr)

    @torch.no_grad()
    def copy_from_hf_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_uses_inverse_sr: bool = True,
    ) -> None:
        """Copy weights from HF-style state_dict keys.

        This works with both bare-model keys, e.g. ``embed.weight``, and CausalLM
        keys, e.g. ``multiscreen.embed.weight``.
        """

        if "multiscreen.embed.weight" in state_dict:
            prefix = "multiscreen."
        elif "embed.weight" in state_dict:
            prefix = ""
        else:
            raise KeyError("state_dict must contain either 'embed.weight' or 'multiscreen.embed.weight'")

        def get(name: str) -> torch.Tensor:
            key = prefix + name
            if key not in state_dict:
                raise KeyError(f"missing state_dict key: {key}")
            return state_dict[key].detach()

        self.W_E.copy_(get("embed.weight"))
        self.s_E.copy_(get("s_E"))
        self.s_F.copy_(get("s_F"))
        for i, layer in enumerate(self.layers):
            layer.copy_from_hf_state_dict(
                state_dict,
                layer_idx=i,
                prefix=prefix,
                hf_uses_inverse_sr=hf_uses_inverse_sr,
            )

    def max_abs_diff_from(
        self,
        other_logits: torch.Tensor,
        input_ids: torch.LongTensor,
        **forward_kwargs: Any,
    ) -> float:
        """Convenience helper for tiny equality tests."""

        oracle_logits = self(input_ids, **forward_kwargs).logits
        return float((oracle_logits - other_logits.to(device=oracle_logits.device)).abs().max().item())




def _get_attr_any(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"object {obj!r} has none of the required attributes: {', '.join(names)}")


def make_oracle_from_hf_model(
    hf_model: Any,
    *,
    hf_uses_inverse_sr: bool = True,
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> PaperMultiscreenForCausalLM:
    """Construct and populate a paper oracle from a HF Multiscreen model.

    Use ``hf_uses_inverse_sr=True`` for the current attached HF implementation,
    which stores the inverse-width parameter.  Set it to False after migrating
    the HF model itself to the paper ``r = sigmoid(s_r)`` parameterization.
    """

    cfg = PaperMultiscreenConfig.from_hf_config(
        hf_model.config,
        **dict(config_overrides or {}),
    )
    oracle = PaperMultiscreenForCausalLM(cfg)
    oracle.copy_from_hf_model(hf_model, hf_uses_inverse_sr=hf_uses_inverse_sr)
    return oracle

__all__ = [
    "PaperMultiscreenConfig",
    "PaperMultiscreenForCausalLM",
    "PaperGatedScreeningLayer",
    "OracleOutput",
    "PositionRule",
    "ComputeDTypeRule",
    "ScreeningCache",
    "unit_normalize",
    "dtype_safe_eps",
    "select_aux_compute_dtype",
    "paper_acceptance_width",
    "trim_relevance",
    "tanh_norm",
    "mipe_gamma",
    "apply_mipe",
    "causal_distance_softmask",
    "make_oracle_from_hf_model",
]
