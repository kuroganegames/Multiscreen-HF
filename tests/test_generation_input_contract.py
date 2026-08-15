"""Stage D cached-generation input layout contracts."""

from __future__ import annotations

import unittest
from typing import Any, Sequence

import torch
from transformers import DynamicCache

from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM
from multiscreen_transformers.configuration_multiscreen import (
    MIPE_POSITION_MODE_PAPER_ABSOLUTE,
    MIPE_POSITION_MODE_REFERENCE,
)


POSITION_MODES = (
    MIPE_POSITION_MODE_PAPER_ABSOLUTE,
    MIPE_POSITION_MODE_REFERENCE,
)


def make_model(*, position_mode: str = MIPE_POSITION_MODE_PAPER_ABSOLUTE) -> MultiscreenForCausalLM:
    torch.manual_seed(20_260_816)
    config = MultiscreenConfig(
        vocab_size=41,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        key_dim=2,
        value_dim=2,
        max_position_embeddings=7,
        mipe_threshold=8.0,
        mipe_position_mode=position_mode,
        mipe_reference_wrap_boundary=4,
        mipe_compute_dtype="fp32",
        softmask_compute_dtype="fp32",
        strict_position_ids=True,
        strict_cache_positions=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    return MultiscreenForCausalLM(config).eval()


@torch.no_grad()
def make_prefix(
    model: MultiscreenForCausalLM,
    input_ids: torch.LongTensor | None = None,
) -> tuple[torch.LongTensor, Sequence[tuple[torch.Tensor, torch.Tensor]]]:
    if input_ids is None:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    if output.past_key_values is None:
        raise AssertionError("use_cache=True returned no cache")
    return input_ids, output.past_key_values


def as_dynamic_cache(
    legacy_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(legacy_cache):
        cache.update(key.clone(), value.clone(), layer_idx=layer_idx)
    return cache


class CachedInputLayoutContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.prefix_ids, self.cache = make_prefix(self.model)
        self.past_length = int(self.prefix_ids.shape[1])

    def prepare(self, input_ids: torch.LongTensor, **kwargs: Any) -> dict[str, Any]:
        return self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=self.cache,
            **kwargs,
        )

    def test_explicit_full_and_suffix_layouts_preserve_every_new_token(self) -> None:
        full_ids = torch.tensor([[1, 2, 3, 11, 12]], dtype=torch.long)
        prepared_full = self.prepare(full_ids, input_ids_include_prefix=True)
        self.assertTrue(torch.equal(prepared_full["input_ids"], full_ids[:, 3:]))
        self.assertEqual(prepared_full["start_pos"], 3)

        for suffix_ids in (
            torch.tensor([[11]], dtype=torch.long),
            torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        ):
            with self.subTest(suffix_length=suffix_ids.shape[1]):
                prepared_suffix = self.prepare(
                    suffix_ids,
                    input_ids_include_prefix=False,
                )
                self.assertTrue(torch.equal(prepared_suffix["input_ids"], suffix_ids))
                self.assertEqual(prepared_suffix["start_pos"], 3)

    def test_explicit_layout_flag_is_strict_bool_and_nonserialized(self) -> None:
        suffix_ids = torch.tensor([[11, 12]], dtype=torch.long)
        for invalid in (0, 1, "false", torch.tensor(False)):
            with self.subTest(value=repr(invalid)), self.assertRaises(TypeError):
                self.prepare(suffix_ids, input_ids_include_prefix=invalid)

        prepared = self.prepare(suffix_ids, input_ids_include_prefix=False)
        self.assertNotIn("input_ids_include_prefix", prepared)
        self.assertNotIn("next_sequence_length", prepared)
        self.assertNotIn("cache_position", prepared)
        self.assertNotIn("input_ids_include_prefix", self.model.config.to_dict())

    def test_existing_positional_arguments_keep_their_original_binding(self) -> None:
        suffix_ids = torch.tensor([[11, 12]], dtype=torch.long)
        attention_mask = torch.ones(
            (1, self.past_length + suffix_ids.shape[1]),
            dtype=torch.long,
        )
        cache_position = torch.arange(
            self.past_length,
            self.past_length + suffix_ids.shape[1],
            dtype=torch.long,
        )
        position_ids = cache_position.unsqueeze(0)

        prepared = self.model.prepare_inputs_for_generation(
            suffix_ids,
            self.cache,
            attention_mask,
            cache_position,
            position_ids,
            self.past_length,
            False,
        )

        self.assertTrue(torch.equal(prepared["input_ids"], suffix_ids))
        self.assertEqual(prepared["start_pos"], self.past_length)
        self.assertIs(prepared["attention_mask"], attention_mask)
        self.assertIs(prepared["use_cache"], False)

    def test_metadata_free_layout_is_only_inferred_when_input_is_shorter_than_cache(self) -> None:
        suffix_ids = torch.tensor([[11, 12]], dtype=torch.long)
        prepared = self.prepare(suffix_ids)
        self.assertTrue(torch.equal(prepared["input_ids"], suffix_ids))

        ambiguous_inputs = (
            torch.tensor([[11, 12, 13]], dtype=torch.long),
            torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        )
        for input_ids in ambiguous_inputs:
            with self.subTest(length=input_ids.shape[1]), self.assertRaisesRegex(
                ValueError, "ambiguous"
            ):
                self.prepare(input_ids)

        with self.assertRaisesRegex(ValueError, "at least one new token"):
            self.prepare(torch.empty((1, 0), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "at least one token after"):
            self.prepare(self.prefix_ids, input_ids_include_prefix=True)

    def test_cache_position_proves_full_or_suffix_layout(self) -> None:
        full_ids = torch.tensor([[1, 2, 3, 11, 12]], dtype=torch.long)
        full = self.prepare(
            full_ids,
            cache_position=torch.tensor([3, 4], dtype=torch.long),
        )
        self.assertTrue(torch.equal(full["input_ids"], full_ids[:, 3:]))

        suffix_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        suffix = self.prepare(
            suffix_ids,
            cache_position=torch.arange(3, 8, dtype=torch.long),
        )
        self.assertTrue(torch.equal(suffix["input_ids"], suffix_ids))

        invalid_positions = (
            torch.tensor([3, 4, 5], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
            torch.tensor([3, 5], dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        for cache_position in invalid_positions:
            with self.subTest(cache_position=cache_position), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.prepare(full_ids, cache_position=cache_position)

    def test_next_sequence_length_proves_full_or_suffix_layout(self) -> None:
        full_ids = torch.tensor([[1, 2, 3, 11, 12]], dtype=torch.long)
        full = self.prepare(full_ids, next_sequence_length=2)
        self.assertTrue(torch.equal(full["input_ids"], full_ids[:, 3:]))

        suffix_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        suffix = self.prepare(suffix_ids, next_sequence_length=5)
        self.assertTrue(torch.equal(suffix["input_ids"], suffix_ids))

        for invalid in (True, 0, -1, 3, 2.0):
            with self.subTest(next_sequence_length=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.prepare(full_ids, next_sequence_length=invalid)

    def test_full_attention_mask_proves_already_sliced_suffix(self) -> None:
        suffix_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        full_mask = torch.ones((1, self.past_length + suffix_ids.shape[1]), dtype=torch.long)
        prepared = self.prepare(suffix_ids, attention_mask=full_mask)
        self.assertTrue(torch.equal(prepared["input_ids"], suffix_ids))
        self.assertIs(prepared["attention_mask"], full_mask)

        same_length_mask = torch.ones_like(suffix_ids)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.prepare(suffix_ids, attention_mask=same_length_mask)

    def test_every_layout_claim_must_agree(self) -> None:
        full_ids = torch.tensor([[1, 2, 3, 11, 12]], dtype=torch.long)
        consistent_full = self.prepare(
            full_ids,
            input_ids_include_prefix=True,
            cache_position=torch.tensor([3, 4], dtype=torch.long),
            next_sequence_length=2,
            attention_mask=torch.ones_like(full_ids),
        )
        self.assertTrue(torch.equal(consistent_full["input_ids"], full_ids[:, 3:]))

        suffix_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        full_mask = torch.ones((1, self.past_length + suffix_ids.shape[1]), dtype=torch.long)
        consistent = self.prepare(
            suffix_ids,
            input_ids_include_prefix=False,
            cache_position=torch.arange(3, 8, dtype=torch.long),
            next_sequence_length=5,
            attention_mask=full_mask,
        )
        self.assertTrue(torch.equal(consistent["input_ids"], suffix_ids))

        contradictory_calls = (
            {
                "input_ids_include_prefix": True,
                "cache_position": torch.arange(3, 8, dtype=torch.long),
            },
            {
                "input_ids_include_prefix": True,
                "next_sequence_length": 5,
            },
            {
                "input_ids_include_prefix": True,
                "attention_mask": full_mask,
            },
            {
                "cache_position": torch.tensor([3, 4], dtype=torch.long),
                "next_sequence_length": 5,
            },
        )
        for kwargs in contradictory_calls:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "disagree"):
                self.prepare(suffix_ids, **kwargs)

    def test_position_ids_validate_after_layout_without_proving_it(self) -> None:
        full_ids = torch.tensor([[1, 2, 3, 11, 12]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.prepare(
                full_ids,
                position_ids=torch.arange(5, dtype=torch.long).unsqueeze(0),
            )

        full_position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)
        prepared_full = self.prepare(
            full_ids,
            input_ids_include_prefix=True,
            position_ids=full_position_ids,
        )
        self.assertTrue(torch.equal(prepared_full["input_ids"], full_ids[:, 3:]))
        self.assertIsNone(prepared_full["position_ids"])

        suffix_position_ids = torch.tensor([[3, 4]], dtype=torch.long)
        self.prepare(
            full_ids,
            input_ids_include_prefix=True,
            position_ids=suffix_position_ids,
        )

        suffix_ids = torch.tensor([[11, 12]], dtype=torch.long)
        full_cached_position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)
        self.prepare(
            suffix_ids,
            input_ids_include_prefix=False,
            position_ids=full_cached_position_ids,
        )

        invalid_position_ids = (
            torch.tensor([[2, 3]], dtype=torch.long),
            torch.arange(4, dtype=torch.long).unsqueeze(0),
        )
        for position_ids in invalid_position_ids:
            with self.subTest(position_ids=position_ids), self.assertRaises(ValueError):
                self.prepare(
                    suffix_ids,
                    input_ids_include_prefix=False,
                    position_ids=position_ids,
                )

    def test_start_pos_validates_cache_origin_but_does_not_prove_layout(self) -> None:
        ambiguous = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.prepare(ambiguous, start_pos=3)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.prepare(
                torch.tensor([[11, 12]], dtype=torch.long),
                start_pos=2,
            )

    def test_legacy_and_dynamic_cache_direct_calls_share_layout_contract(self) -> None:
        suffix_ids = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        legacy = self.prepare(suffix_ids, input_ids_include_prefix=False)
        dynamic = self.model.prepare_inputs_for_generation(
            suffix_ids,
            past_key_values=as_dynamic_cache(self.cache),
            input_ids_include_prefix=False,
        )
        self.assertTrue(torch.equal(dynamic["input_ids"], legacy["input_ids"]))
        self.assertEqual(dynamic["start_pos"], legacy["start_pos"])


class CachedExecutionContractTests(unittest.TestCase):
    def test_full_one_shot_and_chunked_cached_suffix_match_in_both_mipe_modes(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        for position_mode in POSITION_MODES:
            model = make_model(position_mode=position_mode)
            with self.subTest(mode=position_mode), torch.no_grad():
                full = model(input_ids=input_ids, use_cache=True, return_dict=True)
                prefix = model(input_ids=input_ids[:, :3], use_cache=True, return_dict=True)

                full_input_model_inputs = model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=prefix.past_key_values,
                    input_ids_include_prefix=True,
                )
                full_input = model(**full_input_model_inputs, return_dict=True)

                one_shot_inputs = model.prepare_inputs_for_generation(
                    input_ids[:, 3:],
                    past_key_values=prefix.past_key_values,
                    input_ids_include_prefix=False,
                )
                one_shot = model(**one_shot_inputs, return_dict=True)

                first_chunk_inputs = model.prepare_inputs_for_generation(
                    input_ids[:, 3:5],
                    past_key_values=prefix.past_key_values,
                    input_ids_include_prefix=False,
                )
                first_chunk = model(**first_chunk_inputs, return_dict=True)
                second_chunk_inputs = model.prepare_inputs_for_generation(
                    input_ids[:, 5:],
                    past_key_values=first_chunk.past_key_values,
                    input_ids_include_prefix=False,
                )
                second_chunk = model(**second_chunk_inputs, return_dict=True)
                chunked_logits = torch.cat([first_chunk.logits, second_chunk.logits], dim=1)

            torch.testing.assert_close(full_input.logits, full.logits[:, 3:], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(one_shot.logits, full.logits[:, 3:], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(chunked_logits, full.logits[:, 3:], rtol=1e-5, atol=1e-5)

    def test_normal_greedy_generate_remains_flag_free_in_both_mipe_modes(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        for position_mode in POSITION_MODES:
            model = make_model(position_mode=position_mode)
            generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": 3,
                "pad_token_id": 0,
            }
            with self.subTest(mode=position_mode), torch.no_grad():
                cached = model.generate(input_ids, use_cache=True, **generation_kwargs)
                uncached = model.generate(input_ids, use_cache=False, **generation_kwargs)
            self.assertTrue(torch.equal(cached, uncached))

    def test_prefilled_dynamic_cache_generate_with_full_input_matches_baseline(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": 2,
            "pad_token_id": 0,
            "use_cache": True,
        }
        for position_mode in POSITION_MODES:
            model = make_model(position_mode=position_mode)
            with self.subTest(mode=position_mode), torch.no_grad():
                baseline = model.generate(input_ids, **generation_kwargs)
                prefix = model(
                    input_ids=input_ids[:, :3],
                    use_cache=True,
                    return_dict=True,
                )
                resumed = model.generate(
                    input_ids,
                    past_key_values=as_dynamic_cache(prefix.past_key_values),
                    attention_mask=torch.ones_like(input_ids),
                    **generation_kwargs,
                )

            self.assertTrue(torch.equal(resumed, baseline))


if __name__ == "__main__":
    unittest.main()
