"""Self-checks for paper_math_oracle.py.

Run:
    python test_paper_math_oracle_selfcheck.py

These tests do not require transformers. They verify that the oracle runs
end-to-end and that HF-style state-dict conversion is internally consistent.
"""

from __future__ import annotations

import torch

from paper_math_oracle import (
    PaperMultiscreenConfig,
    PaperMultiscreenForCausalLM,
    paper_acceptance_width,
    trim_relevance,
)


def assert_close(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> None:
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = float((a.detach() - b.detach()).abs().max().item())
        raise AssertionError(f"{name} mismatch: max_abs_diff={diff:.6g}\n{exc}") from exc


def to_hf_like_state_dict(model: PaperMultiscreenForCausalLM) -> dict[str, torch.Tensor]:
    """Export oracle weights into the current HF-port state_dict layout."""

    sd: dict[str, torch.Tensor] = {
        "embed.weight": model.W_E.detach().clone(),
        "s_E": model.s_E.detach().clone(),
        "s_F": model.s_F.detach().clone(),
    }
    for i, layer in enumerate(model.layers):
        p = f"layers.{i}.block."
        h = model.config.num_attention_heads
        e = model.config.hidden_size
        dk = model.config.key_dim
        dv = model.config.value_dim
        sd[p + "q_proj.weight"] = layer.W_Q.detach().permute(0, 2, 1).reshape(h * dk, e).clone()
        sd[p + "k_proj.weight"] = layer.W_K.detach().permute(0, 2, 1).reshape(h * dk, e).clone()
        sd[p + "v_proj.weight"] = layer.W_V.detach().permute(0, 2, 1).reshape(h * dv, e).clone()
        sd[p + "g_proj.weight"] = layer.W_G.detach().permute(0, 2, 1).reshape(h * dv, e).clone()
        sd[p + "o_proj.weight"] = layer.W_O.detach().reshape(h * dv, e).transpose(0, 1).contiguous().clone()
        sd[p + "sw"] = layer.s_w.detach().clone()
        # Current HF port uses inverse acceptance width. Since oracle stores paper s_r,
        # the equivalent HF value is -s_r.
        sd[p + "sr"] = (-layer.s_r.detach()).clone()
        sd[p + "sO"] = layer.s_O.detach().clone()
    return sd


def test_end_to_end_smoke() -> None:
    torch.manual_seed(0)
    cfg = PaperMultiscreenConfig(
        vocab_size=17,
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=3,
        key_dim=4,
        value_dim=5,
        max_position_embeddings=16,
    )
    model = PaperMultiscreenForCausalLM(cfg).to(dtype=torch.float64)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.long)
    out = model(input_ids, attention_mask=attention_mask, labels=input_ids, return_aux=True)

    assert out.logits.shape == (2, 5, 17)
    assert out.hidden_states.shape == (2, 5, 12)
    assert out.loss is not None and out.loss.ndim == 0
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.hidden_states).all()
    assert torch.isfinite(out.loss).all()
    assert out.aux is not None and "similarity" in out.aux


def test_sr_parameterization_equivalence_for_hf_current() -> None:
    torch.manual_seed(1)
    sim = torch.rand(2, 4, 3, 3, dtype=torch.float64) * 2.0 - 1.0
    sr_hf_inverse = torch.randn(4, dtype=torch.float64)
    sr_paper = -sr_hf_inverse

    # Paper alpha with s_r_paper = -s_r_hf.
    alpha_paper = trim_relevance(sim, paper_acceptance_width(sr_paper))
    # Directly emulate the current HF-port inverse-width Trim.
    inv_r = torch.exp(sr_hf_inverse).add(1.0).view(1, -1, 1, 1)
    alpha_hf_current = torch.clamp(1.0 - inv_r * (1.0 - sim), min=0.0).square()
    assert_close("paper-vs-current-HF Trim", alpha_paper, alpha_hf_current, atol=1e-12, rtol=1e-12)



def test_sr_parameterization_extreme_values_for_hf_current() -> None:
    sim = torch.linspace(-1.0, 1.0, steps=11, dtype=torch.float64).view(1, 1, 1, 11).expand(1, 5, 1, 11)
    sr_hf_inverse = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0], dtype=torch.float64)
    sr_paper = -sr_hf_inverse
    alpha_paper = trim_relevance(sim, paper_acceptance_width(sr_paper))
    inv_r = torch.exp(sr_hf_inverse).add(1.0).view(1, -1, 1, 1)
    alpha_hf_current = torch.clamp(1.0 - inv_r * (1.0 - sim), min=0.0).square()
    assert_close("paper-vs-current-HF Trim extreme sr", alpha_paper, alpha_hf_current, atol=1e-12, rtol=1e-12)

def test_hf_like_state_dict_round_trip() -> None:
    torch.manual_seed(2)
    cfg = PaperMultiscreenConfig(
        vocab_size=13,
        hidden_size=10,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=3,
        value_dim=4,
        max_position_embeddings=12,
    )
    a = PaperMultiscreenForCausalLM(cfg).to(dtype=torch.float64)
    b = PaperMultiscreenForCausalLM(cfg).to(dtype=torch.float64)
    b.copy_from_hf_state_dict(to_hf_like_state_dict(a), hf_uses_inverse_sr=True)

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    out_a = a(input_ids).logits
    out_b = b(input_ids).logits
    assert_close("round-trip logits", out_b, out_a, atol=0.0, rtol=0.0)


def test_cache_matches_full_context() -> None:
    torch.manual_seed(3)
    cfg = PaperMultiscreenConfig(
        vocab_size=19,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=4,
        value_dim=3,
        max_position_embeddings=16,
    )
    model = PaperMultiscreenForCausalLM(cfg).to(dtype=torch.float64)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    full = model(input_ids).logits

    prefix = model(input_ids[:, :4], use_cache=True)
    suffix = model(input_ids[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)
    assert_close("cache suffix logits", suffix.logits, full[:, 4:, :], atol=1e-10, rtol=1e-10)


def test_cache_position_contract_rejects_misalignment() -> None:
    torch.manual_seed(4)
    cfg = PaperMultiscreenConfig(
        vocab_size=19,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=4,
        value_dim=3,
        max_position_embeddings=16,
        strict_cache_positions=True,
    )
    model = PaperMultiscreenForCausalLM(cfg).to(dtype=torch.float64)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    try:
        model(input_ids, start_pos=2)
    except ValueError as exc:
        assert "start_pos=0" in str(exc)
    else:
        raise AssertionError("nonzero start_pos without cache must fail")

    prefix = model(input_ids[:, :3], use_cache=True)
    try:
        model(input_ids[:, 3:], past_key_values=prefix.past_key_values, start_pos=1, use_cache=True)
    except ValueError as exc:
        assert "past_len=3" in str(exc)
    else:
        raise AssertionError("cached start_pos != past_len must fail")


if __name__ == "__main__":
    test_end_to_end_smoke()
    test_sr_parameterization_equivalence_for_hf_current()
    test_sr_parameterization_extreme_values_for_hf_current()
    test_hf_like_state_dict_round_trip()
    test_cache_matches_full_context()
    test_cache_position_contract_rejects_misalignment()
    print("paper_math_oracle self-checks passed")
