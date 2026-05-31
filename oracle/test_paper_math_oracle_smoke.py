import torch

from paper_math_oracle import PaperMultiscreenConfig, PaperMultiscreenForCausalLM


def tiny_config(**overrides):
    kwargs = dict(
        vocab_size=17,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=4,
        value_dim=3,
        max_position_embeddings=16,
        mipe_threshold=8.0,
    )
    kwargs.update(overrides)
    return PaperMultiscreenConfig(**kwargs)


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    model = PaperMultiscreenForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    labels = torch.randint(0, model.config.vocab_size, (2, 5))
    out = model(input_ids, labels=labels, output_hidden_states=True, return_aux=True)
    assert out.logits.shape == (2, 5, model.config.vocab_size)
    assert out.hidden_states.shape == (2, 5, model.config.hidden_size)
    assert out.loss is not None and out.loss.ndim == 0
    assert out.all_hidden_states is not None and len(out.all_hidden_states) == model.config.num_hidden_layers + 1
    assert out.aux is not None
    assert "relevance" in out.aux
    assert len(out.aux["relevance"]) == model.config.num_hidden_layers


def test_cache_matches_full_pass_next_token_logits():
    torch.manual_seed(1)
    model = PaperMultiscreenForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))

    full = model(input_ids).logits
    prefix = model(input_ids[:, :4], use_cache=True)
    suffix = model(input_ids[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)

    torch.testing.assert_close(suffix.logits, full[:, 4:, :], rtol=1e-5, atol=1e-5)


def test_logits_to_keep():
    torch.manual_seed(2)
    model = PaperMultiscreenForCausalLM(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    full = model(input_ids).logits
    tail = model(input_ids, logits_to_keep=2).logits
    torch.testing.assert_close(tail, full[:, -2:, :], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_forward_shapes_and_loss()
    test_cache_matches_full_pass_next_token_logits()
    test_logits_to_keep()
    print("paper_math_oracle smoke tests passed")
