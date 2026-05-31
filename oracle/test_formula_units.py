import math
import torch

from paper_math_oracle import (
    PaperMultiscreenConfig,
    PaperMultiscreenForCausalLM,
    apply_mipe,
    causal_distance_softmask,
    dtype_safe_eps,
    mipe_gamma,
    paper_acceptance_width,
    unit_normalize,
    tanh_norm,
    trim_relevance,
)


def test_trim_threshold_values():
    # r = 0.5 when s_r = 0.  Similarities <= 0.5 should be zero.
    r = paper_acceptance_width(torch.tensor([0.0]))
    sim = torch.tensor([[[[1.0, 0.75, 0.5, 0.0]]]])
    got = trim_relevance(sim, r)
    expected = torch.tensor([[[[1.0, 0.25, 0.0, 0.0]]]])
    torch.testing.assert_close(got, expected)


def test_softmask_values():
    # For query i=2, window w=3, valid j are 0,1,2 because -3 < j-i <= 0.
    mask = causal_distance_softmask(
        t_new=1,
        t_total=4,
        start_pos=2,
        w=torch.tensor([3.0]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    # rels j-i are [-2, -1, 0, 1].
    expected = torch.tensor([[[[
        0.5 * (math.cos(-2 * math.pi / 3) + 1.0),
        0.5 * (math.cos(-1 * math.pi / 3) + 1.0),
        1.0,
        0.0,
    ]]]])
    torch.testing.assert_close(mask, expected, rtol=1e-6, atol=1e-6)


def test_softmask_boundaries_fractional_and_multihead():
    # Query i=3 over keys j=0..4 gives rel [-3, -2, -1, 0, +1].
    # rel == -w is excluded; rel == 0 is 1; future rel > 0 is 0.
    w = torch.tensor([3.0, 2.5], dtype=torch.float64)
    mask = causal_distance_softmask(
        t_new=1,
        t_total=5,
        start_pos=3,
        w=w,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    rels = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    expected_heads = []
    for wh in w:
        valid = (rels <= 0) & (rels > -wh)
        expected_heads.append(0.5 * (torch.cos(math.pi * rels / wh) + 1.0) * valid)
    expected = torch.stack(expected_heads, dim=0).view(1, 2, 1, 5)
    torch.testing.assert_close(mask, expected, rtol=1e-12, atol=1e-12)
    assert mask[0, 0, 0, 0].item() == 0.0  # rel == -w for head 0
    assert mask[0, :, 0, 3].eq(1.0).all()  # rel == 0
    assert mask[0, :, 0, 4].eq(0.0).all()  # future token


def test_mipe_is_identity_when_w_at_threshold_or_larger():
    q = torch.randn(2, 3, 1, 4)
    k = torch.randn(2, 3, 1, 4)
    w = torch.tensor([8.0])
    q_rot, k_rot = apply_mipe(
        q,
        k,
        w,
        start_pos=0,
        threshold=8.0,
        max_position_embeddings=16,
        position_rule="paper",
    )
    assert mipe_gamma(w, threshold=8.0).item() == 0.0
    torch.testing.assert_close(q_rot, q)
    torch.testing.assert_close(k_rot, k)


def test_mipe_hand_calculation_with_offset_and_extra_dims():
    q = torch.tensor([[[[1.0, 0.0, 3.0, -4.0]], [[0.0, 1.0, 5.0, 6.0]]]])
    k = torch.tensor([[[[0.0, 1.0, 7.0, 8.0]], [[1.0, 0.0, -2.0, 9.0]]]])
    w = torch.tensor([2.0])
    threshold = 8.0
    start_pos = 3
    q_rot, k_rot = apply_mipe(
        q,
        k,
        w,
        start_pos=start_pos,
        threshold=threshold,
        max_position_embeddings=4,
        position_rule="paper",
    )
    gamma = 0.5 * (math.cos(math.pi * 2.0 / threshold) + 1.0)
    for t, pos in enumerate([3, 4]):
        angle = pos * math.pi * gamma / 2.0
        c, s = math.cos(angle), math.sin(angle)
        q0, q1 = q[0, t, 0, 0].item(), q[0, t, 0, 1].item()
        k0, k1 = k[0, t, 0, 0].item(), k[0, t, 0, 1].item()
        torch.testing.assert_close(q_rot[0, t, 0, 0], torch.tensor(q0 * c - q1 * s), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(q_rot[0, t, 0, 1], torch.tensor(q0 * s + q1 * c), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k_rot[0, t, 0, 0], torch.tensor(k0 * c - k1 * s), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k_rot[0, t, 0, 1], torch.tensor(k0 * s + k1 * c), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(q_rot[..., 2:], q[..., 2:])
    torch.testing.assert_close(k_rot[..., 2:], k[..., 2:])


def test_paper_position_rule_does_not_modulo_after_max_position():
    q = torch.tensor([[[[1.0, 0.0, 2.0]], [[1.0, 0.0, 3.0]]]])
    k = torch.tensor([[[[0.0, 1.0, 4.0]], [[0.0, 1.0, 5.0]]]])
    w = torch.tensor([2.0])
    common = dict(w=w, start_pos=4, threshold=8.0, max_position_embeddings=4)
    q_paper, k_paper = apply_mipe(q, k, position_rule="paper", **common)
    q_hf, k_hf = apply_mipe(q, k, position_rule="hf_mod_after_max_position", **common)
    # At absolute position 4, paper uses 4 while HF-compatible mode uses 4 % w = 0.
    assert not torch.allclose(q_paper[..., :2], q_hf[..., :2])
    assert not torch.allclose(k_paper[..., :2], k_hf[..., :2])
    # Extra dimensions remain untouched in both modes.
    torch.testing.assert_close(q_paper[..., 2:], q[..., 2:])
    torch.testing.assert_close(q_hf[..., 2:], q[..., 2:])


def test_tanh_norm_zero_and_direction():
    z = torch.zeros(2, 3)
    torch.testing.assert_close(tanh_norm(z), z)
    x = torch.tensor([[3.0, 4.0]])
    y = tanh_norm(x)
    # Direction preserved: y is a positive scalar multiple of x.
    torch.testing.assert_close(y / x, torch.full_like(x, math.tanh(5.0) / 5.0))


def test_tanh_norm_and_unit_normalize_zero_fp16_are_finite():
    z = torch.zeros(2, 3, dtype=torch.float16)
    assert dtype_safe_eps(z, 1e-12) > 0.0
    zn = tanh_norm(z, eps=1e-8)
    un = torch.zeros_like(z)
    torch.testing.assert_close(zn, un)
    torch.testing.assert_close(tanh_norm(unit_normalize(z, eps=1e-12)), un)
    assert torch.isfinite(zn).all()


def test_cache_position_contract_rejects_misalignment():
    torch.manual_seed(0)
    cfg = PaperMultiscreenConfig(
        vocab_size=11,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        key_dim=4,
        value_dim=3,
        max_position_embeddings=16,
        strict_cache_positions=True,
    )
    model = PaperMultiscreenForCausalLM(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    try:
        model(input_ids, start_pos=1)
    except ValueError as exc:
        assert "start_pos=0" in str(exc)
    else:
        raise AssertionError("nonzero no-cache start_pos should fail")

    prefix = model(input_ids[:, :2], use_cache=True)
    try:
        model(input_ids[:, 2:], past_key_values=prefix.past_key_values, start_pos=0, use_cache=True)
    except ValueError as exc:
        assert "past_len=2" in str(exc)
    else:
        raise AssertionError("cached start_pos != past_len should fail")


def test_low_precision_aux_compute_dtype_modes_are_selectable():
    # This guards the P0-2 reference-compatibility path: in bf16, MiPE can differ
    # between stable fp32 auxiliary math and reference incoming-dtype math near
    # the long-position modulo branch.  Both modes must be explicit and finite.
    import torch
    from paper_math_oracle import apply_mipe

    q = torch.tensor([[[[0.75, -0.50]]]], dtype=torch.bfloat16)
    k = torch.tensor([[[[-0.25, 0.875]]]], dtype=torch.bfloat16)
    w = torch.tensor([3.25], dtype=torch.bfloat16)
    q_fp32, k_fp32 = apply_mipe(
        q,
        k,
        w,
        start_pos=14,
        threshold=8.0,
        max_position_embeddings=12,
        position_rule="hf_mod_after_max_position",
        compute_dtype_rule="fp32",
    )
    q_ref, k_ref = apply_mipe(
        q,
        k,
        w,
        start_pos=14,
        threshold=8.0,
        max_position_embeddings=12,
        position_rule="hf_mod_after_max_position",
        compute_dtype_rule="reference",
    )
    assert torch.isfinite(q_fp32).all()
    assert torch.isfinite(k_fp32).all()
    assert torch.isfinite(q_ref).all()
    assert torch.isfinite(k_ref).all()
    assert q_fp32.dtype == q_ref.dtype == torch.bfloat16
    assert k_fp32.dtype == k_ref.dtype == torch.bfloat16



if __name__ == "__main__":
    test_trim_threshold_values()
    test_softmask_values()
    test_softmask_boundaries_fractional_and_multihead()
    test_mipe_is_identity_when_w_at_threshold_or_larger()
    test_mipe_hand_calculation_with_offset_and_extra_dims()
    test_paper_position_rule_does_not_modulo_after_max_position()
    test_tanh_norm_zero_and_direction()
    test_tanh_norm_and_unit_normalize_zero_fp16_are_finite()
    test_cache_position_contract_rejects_misalignment()
    test_low_precision_aux_compute_dtype_modes_are_selectable()
    print("formula unit tests passed")
