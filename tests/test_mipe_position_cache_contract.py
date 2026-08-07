"""Executable P0.5-C2 MiPE position and contiguous-cache contracts."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoConfig, DynamicCache

from multiscreen_transformers import (
    MultiscreenConfig,
    MultiscreenForCausalLM,
    register_multiscreen_auto_classes,
)
from multiscreen_transformers.modeling_multiscreen import GatedScreeningBlock
from oracle.paper_math_oracle import (
    PaperMultiscreenConfig,
    apply_mipe as oracle_apply_mipe,
    make_oracle_from_hf_model,
    unit_normalize as oracle_unit_normalize,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VENDORED_REFERENCE_ROOT = REPOSITORY_ROOT / "third_party/multiscreen-pytorch"

PAPER_ABSOLUTE = "paper_absolute"
REFERENCE_MOD = "reference_mod_after_wrap_boundary"
POSITION_MODES = (PAPER_ABSOLUTE, REFERENCE_MOD)

POSITION_MATRIX = (0, 1, 255, 256, 257, 4095, 4096, 4097, 8191, 8192, 131_071)
WINDOW_MATRIX = (64.0, 255.5, 256.0, 320.0, 3.25)
WRAP_BOUNDARY = 4096
HISTORICAL_CONFIG_BOUNDARIES = {
    "configs/p0_3_multiscreen_psi8_768/config.json": 128,
    "configs/p0_3_multiscreen_psi16_768/config.json": 128,
    "configs/p0_4_multiscreen_psi8_gpt2_ctx4096/config.json": 4096,
    "configs/p0_4_multiscreen_psi16_gpt2_ctx4096/config.json": 4096,
}


def make_config(
    *,
    position_mode: str = PAPER_ABSOLUTE,
    wrap_boundary: int = WRAP_BOUNDARY,
    max_position_embeddings: int = 256,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 1,
    key_dim: int = 2,
    value_dim: int = 2,
    hidden_size: int = 4,
    mipe_threshold: float = 256.0,
    mipe_compute_dtype: str = "fp32",
    softmask_compute_dtype: str = "fp32",
) -> MultiscreenConfig:
    return MultiscreenConfig(
        vocab_size=37,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        max_position_embeddings=max_position_embeddings,
        mipe_threshold=mipe_threshold,
        mipe_position_mode=position_mode,
        mipe_reference_wrap_boundary=wrap_boundary,
        mipe_compute_dtype=mipe_compute_dtype,
        softmask_compute_dtype=softmask_compute_dtype,
        strict_position_ids=True,
        strict_cache_positions=True,
        use_cache=True,
    )


def make_model(**config_kwargs: Any) -> MultiscreenForCausalLM:
    torch.manual_seed(20_260_807)
    return MultiscreenForCausalLM(make_config(**config_kwargs)).eval()


def apply_oracle_mipe(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    start_pos: int,
    position_rule: str,
    wrap_boundary: int,
    compute_dtype_rule: str,
    threshold: float = 256.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the oracle while retaining compatibility with its pre-C2 helper name.

    The public C2 boundary lives in ``mipe_reference_wrap_boundary``. The oracle's
    pre-C2 function called the same numerical value ``max_position_embeddings``.
    During migration, pass both names when the new keyword is present so this test
    checks the decided semantics without depending on a private refactor order.
    """

    kwargs: dict[str, Any] = {
        "start_pos": start_pos,
        "threshold": threshold,
        "max_position_embeddings": wrap_boundary,
        "position_rule": position_rule,
        "compute_dtype_rule": compute_dtype_rule,
    }
    parameters = inspect.signature(oracle_apply_mipe).parameters
    if "mipe_reference_wrap_boundary" in parameters:
        kwargs["mipe_reference_wrap_boundary"] = wrap_boundary
    elif "reference_wrap_boundary" in parameters:
        kwargs["reference_wrap_boundary"] = wrap_boundary
    return oracle_apply_mipe(q, k, w, **kwargs)


def assert_cache_close(
    testcase: unittest.TestCase,
    actual: Sequence[tuple[torch.Tensor, torch.Tensor]],
    expected: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    testcase.assertEqual(len(actual), len(expected))
    for layer_idx, ((actual_k, actual_v), (expected_k, expected_v)) in enumerate(zip(actual, expected)):
        with testcase.subTest(layer=layer_idx, component="K"):
            torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)
        with testcase.subTest(layer=layer_idx, component="V"):
            torch.testing.assert_close(actual_v, expected_v, rtol=rtol, atol=atol)


def make_prefix_cache(
    config: MultiscreenConfig,
    prefix_length: int,
    *,
    batch_size: int = 1,
    seed: int = 20_260_808,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed + prefix_length)
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(config.num_hidden_layers):
        key = torch.randn(
            batch_size,
            config.num_attention_heads,
            prefix_length,
            config.key_dim,
            generator=generator,
        )
        value = torch.randn(
            batch_size,
            config.num_attention_heads,
            prefix_length,
            config.value_dim,
            generator=generator,
        )
        layers.append((F.normalize(key, dim=-1), F.normalize(value, dim=-1)))
    return tuple(layers)


@torch.no_grad()
def make_one_layer_oracle_prefix_cache(
    oracle: Any,
    input_ids: torch.LongTensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Derive the exact one-layer full-prefill K/V without prefix screening."""

    if len(oracle.layers) != 1:
        raise ValueError("allocation-safe direct prefix construction requires exactly one layer")
    layer = oracle.layers[0]
    hidden_states = oracle.embed_input_ids(input_ids)
    key = torch.einsum("bte,hek->bthk", hidden_states, layer.W_K)
    value = torch.einsum("bte,hev->bthv", hidden_states, layer.W_V)
    key = oracle_unit_normalize(key, eps=oracle.config.norm_eps)
    value = oracle_unit_normalize(value, eps=oracle.config.norm_eps)
    _, key = apply_oracle_mipe(
        torch.zeros_like(key),
        key,
        layer.s_w.exp() + 1.0,
        start_pos=0,
        position_rule=oracle.config.position_rule,
        wrap_boundary=oracle.config.mipe_reference_wrap_boundary,
        compute_dtype_rule=oracle.config.mipe_compute_dtype,
        threshold=oracle.config.mipe_threshold,
    )
    return ((key.transpose(1, 2), value.transpose(1, 2)),)


def run_cached_chunks(
    model: MultiscreenForCausalLM,
    input_ids: torch.LongTensor,
    initial_cache: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
    chunk_sizes: Iterable[int],
) -> tuple[torch.Tensor, Sequence[tuple[torch.Tensor, torch.Tensor]]]:
    cursor = 0
    past_key_values = initial_cache
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk_size in chunk_sizes:
            next_cursor = cursor + chunk_size
            output = model(
                input_ids=input_ids[:, cursor:next_cursor],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits.append(output.logits)
            past_key_values = output.past_key_values
            cursor = next_cursor
    if cursor != input_ids.shape[1]:
        raise AssertionError(f"chunk sizes consumed {cursor} tokens, expected {input_ids.shape[1]}")
    if past_key_values is None:
        raise AssertionError("use_cache=True returned no cache")
    return torch.cat(logits, dim=1), past_key_values


class MipeFormulaContractTests(unittest.TestCase):
    def _block(
        self,
        *,
        position_mode: str,
        compute_dtype: str,
        num_heads: int,
        max_position_embeddings: int = 256,
        wrap_boundary: int = WRAP_BOUNDARY,
    ) -> GatedScreeningBlock:
        config = make_config(
            position_mode=position_mode,
            wrap_boundary=wrap_boundary,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_heads,
            hidden_size=2 * num_heads,
            mipe_compute_dtype=compute_dtype,
        )
        return GatedScreeningBlock(config, layer_idx=0)

    def test_sparse_position_window_matrix_matches_oracle_in_both_modes(self) -> None:
        q = torch.tensor([[[[0.75, -0.50]] * len(WINDOW_MATRIX)]], dtype=torch.float32)
        k = torch.tensor([[[[-0.25, 0.875]] * len(WINDOW_MATRIX)]], dtype=torch.float32)
        windows = torch.tensor(WINDOW_MATRIX, dtype=torch.float32)

        for position_mode in POSITION_MODES:
            block = self._block(
                position_mode=position_mode,
                compute_dtype="fp32",
                num_heads=len(WINDOW_MATRIX),
            )
            for position in POSITION_MATRIX:
                with self.subTest(mode=position_mode, position=position):
                    hf_q, hf_k = block._apply_mipe(q, k, windows, start_pos=position)
                    oracle_q, oracle_k = apply_oracle_mipe(
                        q,
                        k,
                        windows,
                        start_pos=position,
                        position_rule=position_mode,
                        wrap_boundary=WRAP_BOUNDARY,
                        compute_dtype_rule="fp32",
                    )
                    torch.testing.assert_close(hf_q, oracle_q, rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(hf_k, oracle_k, rtol=1e-6, atol=1e-6)

    def test_stable_and_reference_auxiliary_dtype_modes_match_oracle_in_bfloat16(self) -> None:
        windows = torch.tensor([3.25, 64.0], dtype=torch.bfloat16)
        q = torch.tensor([[[[0.75, -0.50], [0.25, 0.625]]]], dtype=torch.bfloat16)
        k = torch.tensor([[[[-0.25, 0.875], [0.50, -0.375]]]], dtype=torch.bfloat16)

        for position_mode in POSITION_MODES:
            for compute_dtype in ("fp32", "reference"):
                block = self._block(
                    position_mode=position_mode,
                    compute_dtype=compute_dtype,
                    num_heads=2,
                )
                for position in (4095, 4096, 4097, 8192, 131_071):
                    with self.subTest(
                        mode=position_mode,
                        compute_dtype=compute_dtype,
                        position=position,
                    ):
                        hf_q, hf_k = block._apply_mipe(q, k, windows, start_pos=position)
                        oracle_q, oracle_k = apply_oracle_mipe(
                            q,
                            k,
                            windows,
                            start_pos=position,
                            position_rule=position_mode,
                            wrap_boundary=WRAP_BOUNDARY,
                            compute_dtype_rule=compute_dtype,
                        )
                        self.assertEqual(hf_q.dtype, torch.bfloat16)
                        self.assertEqual(hf_k.dtype, torch.bfloat16)
                        self.assertTrue(torch.isfinite(hf_q).all())
                        self.assertTrue(torch.isfinite(hf_k).all())
                        torch.testing.assert_close(hf_q, oracle_q, rtol=5e-3, atol=5e-3)
                        torch.testing.assert_close(hf_k, oracle_k, rtol=5e-3, atol=5e-3)

    def test_paper_ignores_wrap_metadata_and_reference_wraps_inclusively(self) -> None:
        q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
        window = torch.tensor([3.25], dtype=torch.float64)

        paper_low_max = self._block(
            position_mode=PAPER_ABSOLUTE,
            compute_dtype="fp32",
            num_heads=1,
            max_position_embeddings=8,
            wrap_boundary=16,
        )
        paper_high_max = self._block(
            position_mode=PAPER_ABSOLUTE,
            compute_dtype="fp32",
            num_heads=1,
            max_position_embeddings=8192,
            wrap_boundary=4096,
        )
        low_q, low_k = paper_low_max._apply_mipe(q, k, window, start_pos=4096)
        high_q, high_k = paper_high_max._apply_mipe(q, k, window, start_pos=4096)
        torch.testing.assert_close(low_q, high_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(low_k, high_k, rtol=0.0, atol=0.0)

        reference_before = self._block(
            position_mode=REFERENCE_MOD,
            compute_dtype="fp32",
            num_heads=1,
            max_position_embeddings=8,
            wrap_boundary=4097,
        )
        reference_at = self._block(
            position_mode=REFERENCE_MOD,
            compute_dtype="fp32",
            num_heads=1,
            max_position_embeddings=8192,
            wrap_boundary=4096,
        )
        before_q, _ = reference_before._apply_mipe(q, k, window, start_pos=4096)
        at_q, _ = reference_at._apply_mipe(q, k, window, start_pos=4096)
        self.assertFalse(torch.allclose(before_q, at_q, rtol=1e-12, atol=1e-12))
        expected_q, _ = apply_oracle_mipe(
            q,
            k,
            window,
            start_pos=4096,
            position_rule=REFERENCE_MOD,
            wrap_boundary=4096,
            compute_dtype_rule="fp32",
        )
        torch.testing.assert_close(at_q, expected_q, rtol=1e-12, atol=1e-12)

    def test_mipe_is_identity_at_and_above_window_threshold(self) -> None:
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        windows = torch.tensor([256.0, 320.0])
        for position_mode in POSITION_MODES:
            block = self._block(
                position_mode=position_mode,
                compute_dtype="fp32",
                num_heads=2,
            )
            with self.subTest(mode=position_mode):
                q_rot, k_rot = block._apply_mipe(q, k, windows, start_pos=131_071)
                torch.testing.assert_close(q_rot, q, rtol=0.0, atol=0.0)
                torch.testing.assert_close(k_rot, k, rtol=0.0, atol=0.0)

    def test_reference_mode_matches_vendored_reference(self) -> None:
        vendor_root = str(VENDORED_REFERENCE_ROOT)
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)
        vendor_config_module = importlib.import_module("multiscreen.config")
        vendor_model_module = importlib.import_module("multiscreen.model")
        vendor_config = vendor_config_module.MultiscreenConfig(
            vocab_size=37,
            hidden_dim=2 * len(WINDOW_MATRIX),
            num_layers=1,
            num_heads=len(WINDOW_MATRIX),
            key_dim=2,
            value_dim=2,
            max_seq_len=WRAP_BOUNDARY,
            mipe_threshold=256.0,
        )
        vendor_block = vendor_model_module.GatedScreeningBlock(vendor_config, layer_idx=0)
        hf_block = self._block(
            position_mode=REFERENCE_MOD,
            compute_dtype="reference",
            num_heads=len(WINDOW_MATRIX),
        )
        q = torch.tensor([[[[0.75, -0.50]] * len(WINDOW_MATRIX)]], dtype=torch.float32)
        k = torch.tensor([[[[-0.25, 0.875]] * len(WINDOW_MATRIX)]], dtype=torch.float32)
        windows = torch.tensor(WINDOW_MATRIX, dtype=torch.float32)

        for position in POSITION_MATRIX:
            with self.subTest(position=position):
                hf_q, hf_k = hf_block._apply_mipe(q, k, windows, start_pos=position)
                ref_q, ref_k = vendor_block._apply_mipe(q, k, windows, start_pos=position)
                torch.testing.assert_close(hf_q, ref_q, rtol=0.0, atol=0.0)
                torch.testing.assert_close(hf_k, ref_k, rtol=0.0, atol=0.0)

    def test_oracle_legacy_aliases_match_canonical_modes(self) -> None:
        q = torch.tensor([[[[0.75, -0.50]]]], dtype=torch.float32)
        k = torch.tensor([[[[-0.25, 0.875]]]], dtype=torch.float32)
        window = torch.tensor([3.25], dtype=torch.float32)
        aliases = {
            "paper": PAPER_ABSOLUTE,
            "hf_mod_after_max_position": REFERENCE_MOD,
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                alias_q, alias_k = apply_oracle_mipe(
                    q,
                    k,
                    window,
                    start_pos=4096,
                    position_rule=alias,
                    wrap_boundary=4096,
                    compute_dtype_rule="fp32",
                )
                canonical_q, canonical_k = apply_oracle_mipe(
                    q,
                    k,
                    window,
                    start_pos=4096,
                    position_rule=canonical,
                    wrap_boundary=4096,
                    compute_dtype_rule="fp32",
                )
                torch.testing.assert_close(alias_q, canonical_q, rtol=0.0, atol=0.0)
                torch.testing.assert_close(alias_k, canonical_k, rtol=0.0, atol=0.0)


class ConfigMigrationContractTests(unittest.TestCase):
    @staticmethod
    def _write_config(directory: Path, data: dict[str, Any]) -> None:
        (directory / "config.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_missing_fields_migrate_to_reference_mode_and_existing_max_boundary(self) -> None:
        source = make_config(
            position_mode=PAPER_ABSOLUTE,
            wrap_boundary=777,
            max_position_embeddings=513,
        ).to_dict()
        source.pop("mipe_position_mode", None)
        source.pop("mipe_reference_wrap_boundary", None)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_config(path, source)
            loaded = MultiscreenConfig.from_pretrained(path)
            self.assertEqual(loaded.mipe_position_mode, REFERENCE_MOD)
            self.assertEqual(loaded.mipe_reference_wrap_boundary, 513)
            self.assertEqual(loaded.max_position_embeddings, 513)
            self.assertEqual(loaded.to_dict()["mipe_position_mode"], REFERENCE_MOD)
            self.assertEqual(loaded.to_dict()["mipe_reference_wrap_boundary"], 513)

    def test_checked_in_p0_configs_remain_unmodified_legacy_fixtures(self) -> None:
        for relative_path, expected_boundary in HISTORICAL_CONFIG_BOUNDARIES.items():
            path = REPOSITORY_ROOT / relative_path
            raw = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(config=relative_path):
                self.assertNotIn("mipe_position_mode", raw)
                self.assertNotIn("mipe_reference_wrap_boundary", raw)
                self.assertEqual(raw["max_position_embeddings"], expected_boundary)

                loaded = MultiscreenConfig.from_pretrained(path.parent)
                self.assertEqual(loaded.mipe_position_mode, REFERENCE_MOD)
                self.assertEqual(loaded.mipe_reference_wrap_boundary, expected_boundary)
                self.assertEqual(loaded.max_position_embeddings, expected_boundary)

    def test_partially_missing_fields_have_deterministic_migration(self) -> None:
        source = make_config(
            position_mode=PAPER_ABSOLUTE,
            wrap_boundary=777,
            max_position_embeddings=513,
        ).to_dict()
        cases = (
            (
                {
                    key: value
                    for key, value in source.items()
                    if key != "mipe_reference_wrap_boundary"
                },
                PAPER_ABSOLUTE,
                513,
            ),
            ({key: value for key, value in source.items() if key != "mipe_position_mode"}, REFERENCE_MOD, 777),
        )
        for data, expected_mode, expected_boundary in cases:
            with self.subTest(mode=expected_mode, boundary=expected_boundary):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    self._write_config(path, data)
                    loaded = MultiscreenConfig.from_pretrained(path)
                    self.assertEqual(loaded.mipe_position_mode, expected_mode)
                    self.assertEqual(loaded.mipe_reference_wrap_boundary, expected_boundary)

    def test_clone_save_load_and_autoconfig_preserve_explicit_mode_and_boundary(self) -> None:
        register_multiscreen_auto_classes()
        for position_mode in POSITION_MODES:
            config = make_config(
                position_mode=position_mode,
                wrap_boundary=4096,
                max_position_embeddings=1024,
            )
            cloned = config.clone(
                mipe_position_mode=position_mode,
                mipe_reference_wrap_boundary=8192,
            )
            self.assertEqual(cloned.mipe_position_mode, position_mode)
            self.assertEqual(cloned.mipe_reference_wrap_boundary, 8192)
            self.assertEqual(config.mipe_reference_wrap_boundary, 4096)

            with self.subTest(mode=position_mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                cloned.save_pretrained(path)
                direct = MultiscreenConfig.from_pretrained(path)
                automatic = AutoConfig.from_pretrained(path)
                for loaded in (direct, automatic):
                    self.assertIsInstance(loaded, MultiscreenConfig)
                    self.assertEqual(loaded.mipe_position_mode, position_mode)
                    self.assertEqual(loaded.mipe_reference_wrap_boundary, 8192)
                    self.assertEqual(loaded.max_position_embeddings, 1024)

    def test_hf_config_rejects_noncanonical_modes_and_invalid_boundaries(self) -> None:
        for invalid_mode in ("paper", "hf_mod_after_max_position", "reference", ""):
            with self.subTest(mode=invalid_mode), self.assertRaises(ValueError):
                make_config(position_mode=invalid_mode)
        for invalid_boundary in (0, -1, 1.5, "4096"):
            with self.subTest(boundary=invalid_boundary), self.assertRaises(ValueError):
                make_config(wrap_boundary=invalid_boundary)  # type: ignore[arg-type]

    def test_oracle_from_hf_config_maps_canonical_and_legacy_configs(self) -> None:
        for position_mode in POSITION_MODES:
            hf_config = make_config(
                position_mode=position_mode,
                wrap_boundary=4096,
                max_position_embeddings=1024,
            )
            oracle_config = PaperMultiscreenConfig.from_hf_config(hf_config)
            with self.subTest(mode=position_mode):
                self.assertEqual(oracle_config.position_rule, position_mode)
                self.assertEqual(oracle_config.mipe_reference_wrap_boundary, 4096)
                self.assertEqual(oracle_config.max_position_embeddings, 1024)

        legacy_data = make_config(
            position_mode=PAPER_ABSOLUTE,
            wrap_boundary=777,
            max_position_embeddings=513,
        ).to_dict()
        legacy_data.pop("mipe_position_mode", None)
        legacy_data.pop("mipe_reference_wrap_boundary", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_config(path, legacy_data)
            legacy_hf = MultiscreenConfig.from_pretrained(path)
            legacy_oracle = PaperMultiscreenConfig.from_hf_config(legacy_hf)
        self.assertEqual(legacy_oracle.position_rule, REFERENCE_MOD)
        self.assertEqual(legacy_oracle.mipe_reference_wrap_boundary, 513)


class StrictPositionAndCacheSchemaTests(unittest.TestCase):
    def test_position_ids_require_integer_batch_shared_nonnegative_contiguous_ranges(self) -> None:
        model = make_model(wrap_boundary=4, max_position_embeddings=7)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        invalid_position_ids = (
            ("non_contiguous", input_ids, torch.tensor([[0, 2, 3]], dtype=torch.long)),
            (
                "batch_specific",
                input_ids.expand(2, -1),
                torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            ),
            (
                "wrong_batch_same_values",
                input_ids,
                torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            ),
            ("negative", input_ids, torch.tensor([[-1, 0, 1]], dtype=torch.long)),
            ("floating", input_ids, torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)),
        )
        for case, case_input_ids, position_ids in invalid_position_ids:
            with self.subTest(case=case), self.assertRaises((TypeError, ValueError)):
                model(input_ids=case_input_ids, position_ids=position_ids, use_cache=False)

        with self.assertRaises((TypeError, ValueError)):
            model(
                input_ids=input_ids,
                position_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                start_pos=0,
                use_cache=False,
            )
        with self.assertRaises((TypeError, ValueError)):
            model(input_ids=input_ids, start_pos=0.5, use_cache=False)

        with torch.no_grad():
            accepted = model(
                input_ids=input_ids,
                position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                use_cache=False,
            )
        self.assertEqual(tuple(accepted.logits.shape), (1, 3, model.config.vocab_size))

    def test_narrow_integer_position_ranges_do_not_wrap(self) -> None:
        model = make_model(wrap_boundary=4, max_position_embeddings=7)
        suffix_ids = torch.tensor([[1, 2]], dtype=torch.long)
        cases = (
            ("uint8", 255, torch.tensor([255, 0], dtype=torch.uint8)),
            ("int8", 127, torch.tensor([127, -128], dtype=torch.int8)),
        )
        for case, prefix_length, cache_position in cases:
            prefix_cache = make_prefix_cache(model.config, prefix_length)
            with self.subTest(case=case, api="forward"), self.assertRaises((TypeError, ValueError)):
                model(
                    input_ids=suffix_ids,
                    position_ids=cache_position.unsqueeze(0),
                    past_key_values=prefix_cache,
                    use_cache=True,
                    return_dict=True,
                )
            with self.subTest(case=case, api="generation"), self.assertRaises((TypeError, ValueError)):
                model.prepare_inputs_for_generation(
                    suffix_ids,
                    past_key_values=prefix_cache,
                    cache_position=cache_position,
                )

    def test_prepare_inputs_validates_entire_cache_position_and_conflicts(self) -> None:
        model = make_model(wrap_boundary=4, max_position_embeddings=7)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        prepared = model.prepare_inputs_for_generation(
            input_ids,
            cache_position=torch.tensor([0, 1, 2], dtype=torch.long),
        )
        self.assertEqual(prepared["start_pos"], 0)
        self.assertTrue(torch.equal(prepared["input_ids"], input_ids))

        invalid_cache_positions = (
            torch.tensor([0, 2, 3], dtype=torch.long),
            torch.tensor([0, 1, 1], dtype=torch.long),
            torch.tensor([-1, 0, 1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        )
        for cache_position in invalid_cache_positions:
            with self.subTest(cache_position=cache_position), self.assertRaises((TypeError, ValueError)):
                model.prepare_inputs_for_generation(input_ids, cache_position=cache_position)

        full_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            prefix = model(input_ids=full_ids[:, :3], use_cache=True, return_dict=True)
        good = model.prepare_inputs_for_generation(
            full_ids,
            past_key_values=prefix.past_key_values,
            cache_position=torch.tensor([3, 4], dtype=torch.long),
        )
        self.assertEqual(good["start_pos"], 3)
        self.assertTrue(torch.equal(good["input_ids"], full_ids[:, 3:]))

        sliced_suffix_ids = torch.tensor([[6, 7, 8, 9, 10]], dtype=torch.long)
        good_sliced = model.prepare_inputs_for_generation(
            sliced_suffix_ids,
            past_key_values=prefix.past_key_values,
            cache_position=torch.tensor([3, 4, 5, 6, 7], dtype=torch.long),
        )
        self.assertEqual(good_sliced["start_pos"], 3)
        self.assertTrue(torch.equal(good_sliced["input_ids"], sliced_suffix_ids))

        bad_kwargs = (
            {"cache_position": torch.tensor([0, 1], dtype=torch.long)},
            {"cache_position": torch.tensor([3, 5], dtype=torch.long)},
            {"cache_position": torch.tensor([3], dtype=torch.long)},
            {"cache_position": torch.tensor([3, 4, 5], dtype=torch.long)},
            {"start_pos": 0},
            {"position_ids": torch.tensor([[0, 1]], dtype=torch.long)},
        )
        for kwargs in bad_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                model.prepare_inputs_for_generation(
                    full_ids,
                    past_key_values=prefix.past_key_values,
                    **kwargs,
                )

    def test_legacy_cache_schema_is_validated_before_screening_math(self) -> None:
        model = make_model(wrap_boundary=4, max_position_embeddings=7)
        config = model.config
        input_ids = torch.tensor([[1]], dtype=torch.long)
        valid = make_prefix_cache(config, 3)
        key, value = valid[0]

        malformed: dict[str, Any] = {
            "wrong_layer_count": (valid[0], valid[0]),
            "not_a_pair": ((key, value, value),),
            "key_value_length_mismatch": ((key, value[:, :, :-1, :]),),
            "wrong_key_rank": ((key[:, 0], value),),
            "wrong_batch": ((key.expand(2, -1, -1, -1), value.expand(2, -1, -1, -1)),),
            "wrong_heads": ((key.expand(-1, 2, -1, -1), value.expand(-1, 2, -1, -1)),),
            "wrong_key_dim": ((torch.zeros(1, 1, 3, 3), value),),
            "wrong_value_dim": ((key, torch.zeros(1, 1, 3, 3)),),
            "wrong_dtype": ((key.to(torch.float64), value.to(torch.float64)),),
        }
        for case, cache in malformed.items():
            with self.subTest(case=case), self.assertRaises((TypeError, ValueError)):
                model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )

        two_layer_model = make_model(
            num_hidden_layers=2,
            wrap_boundary=4,
            max_position_embeddings=7,
        )
        two_layer_cache = make_prefix_cache(two_layer_model.config, 3)
        inconsistent_layers = (
            two_layer_cache[0],
            (
                two_layer_cache[1][0][:, :, :-1, :],
                two_layer_cache[1][1][:, :, :-1, :],
            ),
        )
        with self.assertRaises((TypeError, ValueError)):
            two_layer_model(
                input_ids=input_ids,
                past_key_values=inconsistent_layers,
                use_cache=True,
                return_dict=True,
            )

        with torch.no_grad():
            accepted = model(
                input_ids=input_ids,
                past_key_values=valid,
                use_cache=True,
                return_dict=True,
            )
        self.assertEqual(accepted.past_key_values[0][0].shape[2], 4)

    def test_empty_and_populated_dynamic_cache_match_legacy_behavior(self) -> None:
        model = make_model(wrap_boundary=4, max_position_embeddings=7)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            baseline = model(input_ids=input_ids, use_cache=True, return_dict=True)
            empty_dynamic = model(
                input_ids=input_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
                return_dict=True,
            )
        torch.testing.assert_close(empty_dynamic.logits, baseline.logits, rtol=0.0, atol=0.0)
        assert_cache_close(self, empty_dynamic.past_key_values, baseline.past_key_values, rtol=0.0, atol=0.0)

        legacy_cache = make_prefix_cache(model.config, 3)
        dynamic_cache = DynamicCache()
        dynamic_cache.update(legacy_cache[0][0].clone(), legacy_cache[0][1].clone(), layer_idx=0)
        suffix_ids = torch.tensor([[4, 5]], dtype=torch.long)
        with torch.no_grad():
            legacy_output = model(
                input_ids=suffix_ids,
                past_key_values=legacy_cache,
                use_cache=True,
                return_dict=True,
            )
            dynamic_output = model(
                input_ids=suffix_ids,
                past_key_values=dynamic_cache,
                use_cache=True,
                return_dict=True,
            )
        torch.testing.assert_close(dynamic_output.logits, legacy_output.logits, rtol=0.0, atol=0.0)
        assert_cache_close(self, dynamic_output.past_key_values, legacy_output.past_key_values, rtol=0.0, atol=0.0)

    def test_partial_dynamic_cache_is_rejected_but_preallocated_empty_is_allowed(self) -> None:
        model = make_model(
            num_hidden_layers=2,
            wrap_boundary=4,
            max_position_embeddings=7,
        )
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        try:
            preallocated_empty = DynamicCache(config=model.config)
        except TypeError:
            preallocated_empty = DynamicCache()
        with torch.no_grad():
            baseline = model(input_ids=input_ids, use_cache=True, return_dict=True)
            accepted_empty = model(
                input_ids=input_ids,
                past_key_values=preallocated_empty,
                use_cache=True,
                return_dict=True,
            )
        torch.testing.assert_close(accepted_empty.logits, baseline.logits, rtol=0.0, atol=0.0)
        assert_cache_close(
            self,
            accepted_empty.past_key_values,
            baseline.past_key_values,
            rtol=0.0,
            atol=0.0,
        )

        complete_cache = make_prefix_cache(model.config, 3)
        partial_cache = DynamicCache()
        partial_cache.update(
            complete_cache[1][0].clone(),
            complete_cache[1][1].clone(),
            layer_idx=1,
        )
        with self.assertRaises((TypeError, ValueError)):
            model(
                input_ids=input_ids[:, :1],
                past_key_values=partial_cache,
                use_cache=True,
                return_dict=True,
            )

    def test_cpu_bfloat16_autocast_uses_bfloat16_cache_and_matches_full_forward(self) -> None:
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                autocast_probe = torch.mm(torch.ones(2, 2), torch.ones(2, 2))
        except (RuntimeError, TypeError, NotImplementedError) as exc:
            self.skipTest(f"CPU bfloat16 autocast is unsupported: {exc}")
        if autocast_probe.dtype != torch.bfloat16:
            self.skipTest(
                "CPU bfloat16 autocast did not select bfloat16 for matrix multiplication: "
                f"got {autocast_probe.dtype}"
            )

        model = make_model(
            position_mode=REFERENCE_MOD,
            wrap_boundary=4,
            max_position_embeddings=7,
            mipe_threshold=8.0,
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prefix = model(
                input_ids=input_ids[:, :4],
                use_cache=True,
                return_dict=True,
            )
            suffix = model(
                input_ids=input_ids[:, 4:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            full = model(input_ids=input_ids, use_cache=False, return_dict=True)

        for cache_name, cache in (
            ("prefix", prefix.past_key_values),
            ("suffix", suffix.past_key_values),
        ):
            for layer_idx, (key, value) in enumerate(cache):
                with self.subTest(cache=cache_name, layer=layer_idx):
                    self.assertEqual(key.dtype, torch.bfloat16)
                    self.assertEqual(value.dtype, torch.bfloat16)
        torch.testing.assert_close(
            suffix.logits.float(),
            full.logits[:, 4:].float(),
            rtol=2e-2,
            atol=2e-2,
        )


    def test_cpu_bfloat16_autocast_preserves_float64_cache_for_double_models(self) -> None:
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                autocast_probe = torch.mm(torch.ones(2, 2), torch.ones(2, 2))
        except (RuntimeError, TypeError, NotImplementedError) as exc:
            self.skipTest(f"CPU bfloat16 autocast is unsupported: {exc}")
        if autocast_probe.dtype != torch.bfloat16:
            self.skipTest(
                "CPU bfloat16 autocast did not select bfloat16 for matrix multiplication: "
                f"got {autocast_probe.dtype}"
            )

        model = make_model(
            position_mode=REFERENCE_MOD,
            wrap_boundary=4,
            max_position_embeddings=7,
            mipe_threshold=8.0,
        ).double()
        oracle = make_oracle_from_hf_model(model).double()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        with torch.no_grad():
            hf_prefix = model(input_ids=input_ids[:, :4], use_cache=True, return_dict=True)
            oracle_prefix = oracle(input_ids[:, :4], use_cache=True)
        assert_cache_close(
            self,
            hf_prefix.past_key_values,
            oracle_prefix.past_key_values,
            rtol=1e-9,
            atol=1e-9,
        )

        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            hf_suffix = model(
                input_ids=input_ids[:, 4:],
                past_key_values=hf_prefix.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            oracle_suffix = oracle(
                input_ids[:, 4:],
                past_key_values=oracle_prefix.past_key_values,
                use_cache=True,
            )

        for cache_name, cache in (
            ("hf_prefix", hf_prefix.past_key_values),
            ("oracle_prefix", oracle_prefix.past_key_values),
            ("hf_suffix", hf_suffix.past_key_values),
            ("oracle_suffix", oracle_suffix.past_key_values),
        ):
            for layer_idx, (key, value) in enumerate(cache):
                with self.subTest(cache=cache_name, layer=layer_idx):
                    self.assertEqual(key.dtype, torch.float64)
                    self.assertEqual(value.dtype, torch.float64)
        torch.testing.assert_close(
            hf_suffix.logits,
            oracle_suffix.logits,
            rtol=1e-9,
            atol=1e-9,
        )
        assert_cache_close(
            self,
            hf_suffix.past_key_values,
            oracle_suffix.past_key_values,
            rtol=1e-9,
            atol=1e-9,
        )

class OracleStrictCacheSchemaContractTests(unittest.TestCase):
    def test_oracle_rejects_malformed_cache_and_offsets_before_equation_math(self) -> None:
        hf_model = make_model(wrap_boundary=4, max_position_embeddings=7)
        oracle = make_oracle_from_hf_model(hf_model)
        input_ids = torch.tensor([[1]], dtype=torch.long)
        valid = make_prefix_cache(hf_model.config, 3)
        key, value = valid[0]

        malformed: dict[str, Any] = {
            "wrong_layer_count": (valid[0], valid[0]),
            "not_a_pair": ((key, value, value),),
            "key_value_length_mismatch": ((key, value[:, :, :-1, :]),),
            "wrong_key_rank": ((key[:, 0], value),),
            "wrong_batch": ((key.expand(2, -1, -1, -1), value.expand(2, -1, -1, -1)),),
            "wrong_heads": ((key.expand(-1, 2, -1, -1), value.expand(-1, 2, -1, -1)),),
            "wrong_key_dim": ((torch.zeros(1, 1, 3, 3), value),),
            "wrong_value_dim": ((key, torch.zeros(1, 1, 3, 3)),),
            "wrong_dtype": ((key.to(torch.float64), value.to(torch.float64)),),
        }
        for case, cache in malformed.items():
            with self.subTest(case=case), self.assertRaises((TypeError, ValueError)):
                oracle(input_ids, past_key_values=cache, use_cache=True)

        two_layer_hf = make_model(
            num_hidden_layers=2,
            wrap_boundary=4,
            max_position_embeddings=7,
        )
        two_layer_oracle = make_oracle_from_hf_model(two_layer_hf)
        two_layer_cache = make_prefix_cache(two_layer_hf.config, 3)
        inconsistent_layers = (
            two_layer_cache[0],
            (
                two_layer_cache[1][0][:, :, :-1, :],
                two_layer_cache[1][1][:, :, :-1, :],
            ),
        )
        with self.assertRaises((TypeError, ValueError)):
            two_layer_oracle(
                input_ids,
                past_key_values=inconsistent_layers,
                use_cache=True,
            )

        for start_pos in (0.5, -1):
            with self.subTest(start_pos=start_pos), self.assertRaises((TypeError, ValueError)):
                oracle(input_ids, start_pos=start_pos, use_cache=False)

        with torch.no_grad():
            accepted = oracle(input_ids, past_key_values=valid, use_cache=True)
        self.assertEqual(accepted.past_key_values[0][0].shape[2], 4)


class CacheEquivalenceContractTests(unittest.TestCase):
    def test_greedy_generation_with_cache_crosses_small_boundary_in_both_modes(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        for position_mode in POSITION_MODES:
            model = make_model(
                position_mode=position_mode,
                wrap_boundary=4,
                max_position_embeddings=7,
                mipe_threshold=8.0,
            )
            generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": 3,
                "pad_token_id": 0,
            }
            with self.subTest(mode=position_mode), torch.no_grad():
                cached = model.generate(input_ids, use_cache=True, **generation_kwargs)
                uncached = model.generate(input_ids, use_cache=False, **generation_kwargs)
            self.assertEqual(tuple(cached.shape), (1, 6))
            self.assertTrue(torch.equal(cached, uncached))

    def test_actual_small_model_full_cache_and_oracle_agree_across_boundary(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
        for position_mode in POSITION_MODES:
            model = make_model(
                position_mode=position_mode,
                wrap_boundary=4,
                max_position_embeddings=7,
                mipe_threshold=8.0,
            )
            oracle = make_oracle_from_hf_model(model)
            with self.subTest(mode=position_mode), torch.no_grad():
                full = model(input_ids=input_ids, use_cache=True, return_dict=True)
                oracle_full = oracle(input_ids, use_cache=True)
                prefix = model(input_ids=input_ids[:, :3], use_cache=True, return_dict=True)
                oracle_prefix_cache = make_one_layer_oracle_prefix_cache(oracle, input_ids[:, :3])
                suffix = model(
                    input_ids=input_ids[:, 3:],
                    past_key_values=oracle_prefix_cache,
                    use_cache=True,
                    return_dict=True,
                )

            torch.testing.assert_close(full.logits, oracle_full.logits, rtol=1e-5, atol=1e-5)
            assert_cache_close(self, oracle_prefix_cache, prefix.past_key_values)
            torch.testing.assert_close(suffix.logits, full.logits[:, 3:], rtol=1e-5, atol=1e-5)
            assert_cache_close(self, suffix.past_key_values, full.past_key_values)

            chunked_logits, chunked_cache = run_cached_chunks(
                model,
                input_ids,
                initial_cache=None,
                chunk_sizes=(3, 2, 1, 3),
            )
            torch.testing.assert_close(chunked_logits, full.logits, rtol=1e-5, atol=1e-5)
            assert_cache_close(self, chunked_cache, full.past_key_values)

    def test_allocation_safe_real_position_full_suffix_oracle_and_chunks_agree(self) -> None:
        scenarios = (
            (4080, 32, (16, 16)),
            (4096, 16, (5, 7, 4)),
            (8192, 16, (3, 5, 8)),
            (4094, 4, (1, 1, 1, 1)),
            (4089, 23, (3, 7, 2, 11)),
        )
        for position_mode in POSITION_MODES:
            model = make_model(
                position_mode=position_mode,
                wrap_boundary=4096,
                max_position_embeddings=256,
            )
            oracle = make_oracle_from_hf_model(model)
            for prefix_length, suffix_length, chunks in scenarios:
                with self.subTest(
                    mode=position_mode,
                    prefix=prefix_length,
                    suffix=suffix_length,
                    chunks=chunks,
                ):
                    # The largest screening matrix is suffix x (prefix + suffix),
                    # not a dense (prefix + suffix)^2 prefill.
                    self.assertLessEqual(suffix_length * (prefix_length + suffix_length), 140_000)
                    full_input_ids = (
                        torch.arange(prefix_length + suffix_length, dtype=torch.long).unsqueeze(0)
                        % model.config.vocab_size
                    )
                    prefix_cache = make_one_layer_oracle_prefix_cache(
                        oracle,
                        full_input_ids[:, :prefix_length],
                    )
                    input_ids = full_input_ids[:, prefix_length:]
                    with torch.no_grad():
                        full_suffix = model(
                            input_ids=input_ids,
                            past_key_values=prefix_cache,
                            use_cache=True,
                            return_dict=True,
                        )
                        oracle_full_suffix = oracle(
                            input_ids,
                            past_key_values=prefix_cache,
                            use_cache=True,
                        )
                    chunked_logits, chunked_cache = run_cached_chunks(
                        model,
                        input_ids,
                        initial_cache=prefix_cache,
                        chunk_sizes=chunks,
                    )
                    # This oracle full-suffix call is equivalent to the suffix
                    # rows of a dense one-layer full-context forward, while the
                    # direct prefix construction avoids its quadratic matrix.
                    torch.testing.assert_close(
                        full_suffix.logits,
                        oracle_full_suffix.logits,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                    assert_cache_close(
                        self,
                        full_suffix.past_key_values,
                        oracle_full_suffix.past_key_values,
                    )
                    torch.testing.assert_close(
                        chunked_logits,
                        oracle_full_suffix.logits,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                    assert_cache_close(self, chunked_cache, oracle_full_suffix.past_key_values)


if __name__ == "__main__":
    unittest.main()
