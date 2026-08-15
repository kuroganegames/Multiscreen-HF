"""Executable P0.5-C1 architecture and all-scale contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from multiscreen_transformers import (
    MultiscreenConfig,
    MultiscreenForCausalLM,
    register_multiscreen_auto_classes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "docs/validation_results/P0_5_C1_ARCHITECTURE_MANIFEST.json"

VOCAB_SIZE = 50_257
KEY_DIM = 16
VALUE_DIM = 64
MIPE_THRESHOLD = 256.0
PAPER_COUNTS = {
    8: (4_134_146, 917_698, 67),
    16: (27_546_626, 14_680_834, 131),
    32: (286_347_266, 234_884_098, 259),
    48: (1_304_884_226, 1_189_092_098, 387),
    64: (3_963_961_346, 3_758_108_674, 515),
}


def independently_derived_shapes(psi: int) -> dict[str, tuple[int, ...]]:
    """Derive expected HF storage shapes directly from the paper matrices."""

    hidden_size = psi * psi
    shapes: dict[str, tuple[int, ...]] = {
        "multiscreen.embed.weight": (VOCAB_SIZE, hidden_size),
        "multiscreen.s_E": (),
        "multiscreen.s_F": (),
    }
    for layer_idx in range(psi):
        prefix = f"multiscreen.layers.{layer_idx}.block"
        shapes[f"{prefix}.q_proj.weight"] = (psi * KEY_DIM, hidden_size)
        shapes[f"{prefix}.k_proj.weight"] = (psi * KEY_DIM, hidden_size)
        shapes[f"{prefix}.v_proj.weight"] = (psi * VALUE_DIM, hidden_size)
        shapes[f"{prefix}.g_proj.weight"] = (psi * VALUE_DIM, hidden_size)
        shapes[f"{prefix}.o_proj.weight"] = (hidden_size, psi * VALUE_DIM)
        shapes[f"{prefix}.sw"] = (psi,)
        shapes[f"{prefix}.sr"] = (psi,)
        shapes[f"{prefix}.sO"] = (psi,)
    return shapes


def independently_derived_counts(psi: int) -> tuple[int, int]:
    """Use named paper shapes, including all learned scalar/vector values."""

    hidden_size = psi * psi
    embedding = VOCAB_SIZE * hidden_size
    per_tile = hidden_size * (2 * KEY_DIM + 3 * VALUE_DIM) + 3
    non_embedding = psi * psi * per_tile + 2
    return embedding + non_embedding, non_embedding


class PaperArchitectureContractTests(unittest.TestCase):
    def test_all_paper_scales_match_independent_shapes_and_counts_on_meta(self) -> None:
        for psi, (paper_total, paper_non_embedding, paper_key_count) in PAPER_COUNTS.items():
            with self.subTest(psi=psi):
                config = MultiscreenConfig.from_psi(
                    psi,
                    vocab_size=VOCAB_SIZE,
                    max_seq_len=256,
                    key_dim=KEY_DIM,
                    value_dim=VALUE_DIM,
                    mipe_threshold=MIPE_THRESHOLD,
                )
                self.assertEqual(config.num_hidden_layers, psi)
                self.assertEqual(config.num_attention_heads, psi)
                self.assertEqual(config.hidden_size, psi * psi)
                self.assertEqual(config.key_dim, KEY_DIM)
                self.assertEqual(config.value_dim, VALUE_DIM)
                self.assertEqual(config.mipe_threshold, MIPE_THRESHOLD)
                self.assertEqual(config.vocab_size, VOCAB_SIZE)

                derived_total, derived_non_embedding = independently_derived_counts(psi)
                self.assertEqual((derived_total, derived_non_embedding), (paper_total, paper_non_embedding))

                with torch.device("meta"):
                    model = MultiscreenForCausalLM(config)

                self.assertTrue(all(parameter.device.type == "meta" for parameter in model.parameters()))
                self.assertTrue(all(buffer.device.type == "meta" for buffer in model.buffers()))
                state_dict = model.state_dict()
                self.assertTrue(all(tensor.device.type == "meta" for tensor in state_dict.values()))

                expected_shapes = independently_derived_shapes(psi)
                actual_shapes = {key: tuple(tensor.shape) for key, tensor in state_dict.items()}
                self.assertEqual(actual_shapes, expected_shapes)
                self.assertEqual(len(actual_shapes), paper_key_count)

                actual_total = sum(parameter.numel() for parameter in model.parameters())
                embedding_count = model.multiscreen.embed.weight.numel()
                self.assertEqual(actual_total, paper_total)
                self.assertEqual(actual_total - embedding_count, paper_non_embedding)
                self.assertEqual(sum(tensor.numel() for tensor in state_dict.values()), paper_total)
                self.assertFalse(any(name.startswith("lm_head.") for name, _ in model.named_parameters()))

    def test_aliases_from_psi_clone_and_conflicts(self) -> None:
        alias_config = MultiscreenConfig(
            vocab_size=101,
            hidden_dim=81,
            num_layers=3,
            num_heads=5,
            key_dim=4,
            value_dim=7,
            max_seq_len=99,
        )
        self.assertEqual((alias_config.hidden_size, alias_config.hidden_dim), (81, 81))
        self.assertEqual((alias_config.num_hidden_layers, alias_config.num_layers), (3, 3))
        self.assertEqual((alias_config.num_attention_heads, alias_config.num_heads), (5, 5))
        self.assertEqual((alias_config.max_position_embeddings, alias_config.max_seq_len), (99, 99))

        for psi in PAPER_COUNTS:
            with self.subTest(from_psi=psi):
                paper_config = MultiscreenConfig.from_psi(psi)
                self.assertEqual(
                    (
                        paper_config.hidden_size,
                        paper_config.num_hidden_layers,
                        paper_config.num_attention_heads,
                        paper_config.key_dim,
                        paper_config.value_dim,
                        paper_config.mipe_threshold,
                        paper_config.vocab_size,
                    ),
                    (psi * psi, psi, psi, KEY_DIM, VALUE_DIM, MIPE_THRESHOLD, VOCAB_SIZE),
                )

        cloned = alias_config.clone(hidden_dim=100, num_layers=4, num_heads=2, max_seq_len=128)
        self.assertEqual(
            (
                cloned.hidden_size,
                cloned.num_hidden_layers,
                cloned.num_attention_heads,
                cloned.max_position_embeddings,
            ),
            (100, 4, 2, 128),
        )
        self.assertEqual(alias_config.hidden_size, 81)

        conflict_cases = [
            {"hidden_size": 64, "hidden_dim": 65},
            {"num_hidden_layers": 2, "num_layers": 3},
            {"num_attention_heads": 2, "num_heads": 3},
            {"max_position_embeddings": 16, "max_seq_len": 17},
        ]
        for kwargs in conflict_cases:
            with self.subTest(conflict=kwargs), self.assertRaises(ValueError):
                MultiscreenConfig(**kwargs)
        with self.assertRaises(ValueError):
            alias_config.clone(hidden_size=64, hidden_dim=65)
        with self.assertRaises(ValueError):
            MultiscreenConfig(tie_word_embeddings=False)

    def test_config_model_save_load_and_autoclass_metadata(self) -> None:
        config = MultiscreenConfig(
            vocab_size=47,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            key_dim=4,
            value_dim=8,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        expected_auto_map = {
            "AutoConfig": "configuration_multiscreen.MultiscreenConfig",
            "AutoModel": "modeling_multiscreen.MultiscreenModel",
            "AutoModelForCausalLM": "modeling_multiscreen.MultiscreenForCausalLM",
        }
        self.assertEqual(config.model_type, "multiscreen")
        self.assertEqual(config.auto_map, expected_auto_map)
        self.assertEqual(config.architectures, ["MultiscreenForCausalLM"])

        torch.manual_seed(91)
        model = MultiscreenForCausalLM(config)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_path = Path(first_dir)
            second_path = Path(second_dir)
            config.save_pretrained(first_path)
            config.save_pretrained(second_path)
            self.assertEqual(
                (first_path / "config.json").read_bytes(),
                (second_path / "config.json").read_bytes(),
            )

            loaded_config = MultiscreenConfig.from_pretrained(first_path)
            for name in (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "key_dim",
                "value_dim",
                "max_position_embeddings",
                "tie_word_embeddings",
                "auto_map",
                "architectures",
            ):
                self.assertEqual(getattr(loaded_config, name), getattr(config, name))

            model.save_pretrained(first_path, safe_serialization=True)
            directly_loaded = MultiscreenForCausalLM.from_pretrained(first_path)
            for key, expected_tensor in model.state_dict().items():
                torch.testing.assert_close(
                    directly_loaded.state_dict()[key],
                    expected_tensor,
                    rtol=0.0,
                    atol=0.0,
                )

            register_multiscreen_auto_classes()
            auto_config = AutoConfig.from_pretrained(first_path)
            self.assertIsInstance(auto_config, MultiscreenConfig)
            auto_model = AutoModelForCausalLM.from_pretrained(first_path)
            self.assertIsInstance(auto_model, MultiscreenForCausalLM)

    def test_normalized_tied_embedding_identity_and_parameter_free_head(self) -> None:
        config = MultiscreenConfig(
            vocab_size=31,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            key_dim=4,
            value_dim=8,
        )
        model = MultiscreenForCausalLM(config)
        self.assertIs(model.get_output_embeddings(), model.lm_head)
        self.assertIs(model.get_input_embeddings(), model.multiscreen.embed)
        self.assertEqual(list(model.lm_head.named_parameters()), [])
        self.assertFalse(any(name.startswith("lm_head.") for name in model.state_dict()))
        self.assertNotIsInstance(model.lm_head.weight, torch.nn.Parameter)

        expected_weight = F.normalize(model.multiscreen.embed.weight, dim=-1) * model.multiscreen.s_F.exp()
        torch.testing.assert_close(model.lm_head.weight, expected_weight, rtol=0.0, atol=0.0)
        hidden_states = torch.randn(2, 3, config.hidden_size)
        torch.testing.assert_close(
            model._compute_logits(hidden_states),
            F.linear(hidden_states, expected_weight),
            rtol=0.0,
            atol=0.0,
        )

    def test_checked_manifest_records_the_independent_contract(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source"]["arxiv_id"], "2604.01178v3")
        self.assertEqual(
            manifest["construction"],
            {
                "max_position_embeddings": 256,
                "position_semantics_contract": False,
            },
        )
        self.assertEqual([record["psi"] for record in manifest["scales"]], list(PAPER_COUNTS))
        for record in manifest["scales"]:
            psi = record["psi"]
            paper_total, paper_non_embedding, paper_key_count = PAPER_COUNTS[psi]
            accounting = record["parameter_accounting"]
            self.assertEqual(accounting["total_parameters"], paper_total)
            self.assertEqual(accounting["non_embedding_parameters"], paper_non_embedding)
            self.assertEqual(record["state_dict"]["key_count"], paper_key_count)
            self.assertEqual(record["allocation"]["device"], "meta")
            self.assertEqual(record["allocation"]["real_weight_elements_allocated"], 0)


if __name__ == "__main__":
    unittest.main()
