"""Focused Stage B contracts for checkpointed caches and zero-valid loss."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM


def make_model() -> MultiscreenForCausalLM:
    torch.manual_seed(20_260_815)
    config = MultiscreenConfig(
        vocab_size=37,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=4,
        value_dim=8,
        max_position_embeddings=32,
        mipe_compute_dtype="fp32",
        softmask_compute_dtype="fp32",
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return MultiscreenForCausalLM(config)


def make_legacy_cache(model: MultiscreenForCausalLM):
    with torch.no_grad():
        output = model.eval()(
            input_ids=torch.tensor([[1, 4]], dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    if output.past_key_values is None:
        raise AssertionError("prefill did not return a cache")
    return output.past_key_values


class GradientCheckpointedCacheContractTests(unittest.TestCase):
    def _assert_rejected_before_layer(
        self,
        *,
        cache,
        alias: bool = False,
        use_cache: bool = False,
    ) -> None:
        model = make_model()
        model.train()
        model.gradient_checkpointing_enable({"use_reentrant": False})
        layer_called = False

        def mark_layer_called(_module, _args):
            nonlocal layer_called
            layer_called = True

        handle = model.multiscreen.layers[0].register_forward_pre_hook(mark_layer_called)
        kwargs = {"kv_caches" if alias else "past_key_values": cache}
        try:
            with self.assertRaisesRegex(
                ValueError,
                "gradient-checkpointed training with past_key_values is unsupported",
            ):
                model(
                    input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long),
                    use_cache=use_cache,
                    return_dict=True,
                    **kwargs,
                )
        finally:
            handle.remove()
        self.assertFalse(layer_called)

    def test_nonempty_legacy_tuple_list_and_alias_fail_fast(self) -> None:
        source = make_model()
        cache = make_legacy_cache(source)
        self._assert_rejected_before_layer(cache=cache)
        self._assert_rejected_before_layer(cache=cache, use_cache=True)
        self._assert_rejected_before_layer(cache=list(cache))
        self._assert_rejected_before_layer(cache=cache, alias=True)

    def test_nonempty_dynamic_cache_fails_fast(self) -> None:
        source = make_model()
        legacy = make_legacy_cache(source)
        dynamic = DynamicCache(legacy)
        self._assert_rejected_before_layer(cache=dynamic)

    def test_supported_checkpoint_and_cache_paths_still_work(self) -> None:
        input_ids = torch.tensor([[1, 5, 7, 2]], dtype=torch.long)
        labels = input_ids.clone()

        checkpointed = make_model().train()
        checkpointed.gradient_checkpointing_enable({"use_reentrant": False})
        output = checkpointed(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()

        cached = make_model()
        prefix_cache = make_legacy_cache(cached)
        with torch.no_grad():
            decoded = cached.eval()(
                input_ids=torch.tensor([[8, 9]], dtype=torch.long),
                past_key_values=prefix_cache,
                use_cache=True,
                return_dict=True,
            )
        self.assertEqual(decoded.logits.shape, (1, 2, cached.config.vocab_size))
        self.assertIsNotNone(decoded.past_key_values)

        plain_training = make_model().train()
        plain_cache = make_legacy_cache(plain_training)
        plain_training.train()
        plain = plain_training(
            input_ids=torch.tensor([[8, 9]], dtype=torch.long),
            past_key_values=plain_cache,
            use_cache=False,
            return_dict=True,
        )
        self.assertTrue(torch.isfinite(plain.logits).all())

    def test_empty_cache_variants_are_allowed(self) -> None:
        for cache in ((), [], DynamicCache()):
            with self.subTest(cache_type=type(cache).__name__):
                model = make_model().train()
                model.gradient_checkpointing_enable({"use_reentrant": False})
                output = model(
                    input_ids=torch.tensor([[1, 5, 7]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=False,
                    return_dict=True,
                )
                self.assertTrue(torch.isfinite(output.logits).all())


class ZeroValidTargetLossContractTests(unittest.TestCase):
    def _assert_graph_zero(
        self,
        *,
        input_ids: list[list[int]],
        labels: list[list[int]],
        attention_mask: list[list[int]] | None,
        labels_are_shifted: bool,
    ) -> None:
        model = make_model().train()
        output = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            attention_mask=(
                None if attention_mask is None else torch.tensor(attention_mask, dtype=torch.long)
            ),
            labels_are_shifted=labels_are_shifted,
            use_cache=False,
            return_dict=True,
        )
        self.assertEqual(float(output.loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(output.loss.requires_grad)
        output.loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)

    def test_all_ignore_shifted_and_unshifted_are_graph_zeros(self) -> None:
        for shifted in (False, True):
            with self.subTest(labels_are_shifted=shifted):
                self._assert_graph_zero(
                    input_ids=[[1, 4, 7]],
                    labels=[[-100, -100, -100]],
                    attention_mask=None,
                    labels_are_shifted=shifted,
                )

    def test_all_masked_shifted_and_unshifted_are_graph_zeros(self) -> None:
        for shifted in (False, True):
            with self.subTest(labels_are_shifted=shifted):
                self._assert_graph_zero(
                    input_ids=[[1, 4, 7]],
                    labels=[[1, 4, 7]],
                    attention_mask=[[0, 0, 0]],
                    labels_are_shifted=shifted,
                )

    def test_sequence_length_one_edges_are_graph_zeros(self) -> None:
        self._assert_graph_zero(
            input_ids=[[1]],
            labels=[[1]],
            attention_mask=None,
            labels_are_shifted=False,
        )
        self._assert_graph_zero(
            input_ids=[[1]],
            labels=[[-100]],
            attention_mask=None,
            labels_are_shifted=True,
        )
        self._assert_graph_zero(
            input_ids=[[1, 4, 7]],
            labels=[[5, -100, -100]],
            attention_mask=None,
            labels_are_shifted=False,
        )

    def test_shifted_sequence_length_one_valid_target_keeps_cross_entropy(self) -> None:
        model = make_model().train()
        output = model(
            input_ids=torch.tensor([[1]], dtype=torch.long),
            labels=torch.tensor([[5]], dtype=torch.long),
            labels_are_shifted=True,
            use_cache=False,
            return_dict=True,
        )
        expected = F.cross_entropy(output.logits[:, 0, :], torch.tensor([5]))
        torch.testing.assert_close(output.loss, expected, rtol=0.0, atol=0.0)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()

    def test_valid_target_nan_is_not_hidden(self) -> None:
        model = make_model().train()
        original_compute_logits = model._compute_logits

        def nan_logits(hidden_states: torch.Tensor) -> torch.Tensor:
            return original_compute_logits(hidden_states) * float("nan")

        model._compute_logits = nan_logits
        output = model(
            input_ids=torch.tensor([[1, 5]], dtype=torch.long),
            labels=torch.tensor([[5, 7]], dtype=torch.long),
            labels_are_shifted=True,
            use_cache=False,
            return_dict=True,
        )
        self.assertTrue(torch.isnan(output.loss))

    def test_mixed_ignore_and_padding_keep_the_existing_cross_entropy(self) -> None:
        cases = (
            ([[0, 0, 1, 5, 7]], [[-100, -100, 1, 5, 7]], [[0, 0, 1, 1, 1]], False),
            ([[1, 5, 7, 0, 0]], [[1, 5, 7, -100, -100]], [[1, 1, 1, 0, 0]], False),
            ([[1, 5, 7, 2]], [[5, -100, 2, -100]], [[1, 1, 1, 1]], True),
        )
        for input_ids, labels, attention_mask, shifted in cases:
            with self.subTest(labels_are_shifted=shifted, input_ids=input_ids):
                model = make_model().train()
                ids = torch.tensor(input_ids, dtype=torch.long)
                targets = torch.tensor(labels, dtype=torch.long)
                mask = torch.tensor(attention_mask, dtype=torch.long)
                output = model(
                    input_ids=ids,
                    labels=targets,
                    attention_mask=mask,
                    labels_are_shifted=shifted,
                    use_cache=False,
                    return_dict=True,
                )
                masked_targets = targets.masked_fill(mask == 0, -100)
                if shifted:
                    expected_logits = output.logits
                    expected_targets = masked_targets
                else:
                    expected_logits = output.logits[..., :-1, :]
                    expected_targets = masked_targets[..., 1:].clone()
                    valid_shift = (mask[..., :-1] != 0) & (mask[..., 1:] != 0)
                    expected_targets.masked_fill_(~valid_shift, -100)
                expected = F.cross_entropy(
                    expected_logits.reshape(-1, model.config.vocab_size),
                    expected_targets.reshape(-1),
                    ignore_index=-100,
                )
                torch.testing.assert_close(output.loss, expected, rtol=0.0, atol=0.0)
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                self.assertTrue(
                    all(
                        parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                )


if __name__ == "__main__":
    unittest.main()
