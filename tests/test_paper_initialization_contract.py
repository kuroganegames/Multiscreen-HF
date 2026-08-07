"""Executable P0.5-C1 initialization contracts."""

from __future__ import annotations

import math
import unittest
from collections import defaultdict
from unittest import mock

import torch
import torch.nn as nn

from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM


class PaperInitializationContractTests(unittest.TestCase):
    def test_requested_random_initializer_arguments_by_parameter_identity(self) -> None:
        config = MultiscreenConfig(
            vocab_size=257,
            hidden_size=81,
            num_hidden_layers=3,
            num_attention_heads=5,
            key_dim=7,
            value_dim=11,
            initializer_range=0.1,
        )
        original_normal = nn.init.normal_
        calls: list[dict[str, object]] = []

        def audited_normal(
            tensor: torch.Tensor,
            mean: float = 0.0,
            std: float = 1.0,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            calls.append(
                {
                    "tensor": tensor,
                    "mean": float(mean),
                    "std": float(std),
                }
            )
            return original_normal(tensor, mean=mean, std=std, generator=generator)

        torch.manual_seed(17)
        with mock.patch.object(nn.init, "normal_", side_effect=audited_normal):
            model = MultiscreenForCausalLM(config)

        names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
        calls_by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
        for call in calls:
            tensor = call["tensor"]
            self.assertIsInstance(tensor, torch.Tensor)
            name = names_by_identity.get(id(tensor))
            self.assertIsNotNone(name, "normal_ initialized a tensor not registered by the final model")
            calls_by_name[name].append(call)  # type: ignore[index]

        expected_std: dict[str, float] = {
            "multiscreen.embed.weight": config.initializer_range / math.sqrt(config.hidden_size)
        }
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"multiscreen.layers.{layer_idx}.block"
            expected_std[f"{prefix}.q_proj.weight"] = config.initializer_range / math.sqrt(config.key_dim)
            expected_std[f"{prefix}.k_proj.weight"] = config.initializer_range / math.sqrt(config.key_dim)
            expected_std[f"{prefix}.v_proj.weight"] = config.initializer_range / math.sqrt(config.value_dim)
            expected_std[f"{prefix}.g_proj.weight"] = config.initializer_range
            expected_std[f"{prefix}.o_proj.weight"] = config.initializer_range / math.sqrt(
                config.hidden_size
            )

        self.assertEqual(set(calls_by_name), set(expected_std))
        for name, requested_std in expected_std.items():
            with self.subTest(parameter=name):
                final_call = calls_by_name[name][-1]
                self.assertEqual(final_call["mean"], 0.0)
                self.assertEqual(final_call["std"], requested_std)
                if name == "multiscreen.embed.weight":
                    self.assertGreaterEqual(len(calls_by_name[name]), 1)
                else:
                    self.assertEqual(len(calls_by_name[name]), 1)

    def test_exact_scalar_and_vector_initializers(self) -> None:
        configs = {
            "paper_psi8": MultiscreenConfig.from_psi(8, vocab_size=257),
            "diagnostic_noncoincident": MultiscreenConfig(
                vocab_size=257,
                hidden_size=81,
                num_hidden_layers=3,
                num_attention_heads=5,
                key_dim=7,
                value_dim=11,
                mipe_threshold=123.0,
            ),
        }
        for case, config in configs.items():
            with self.subTest(case=case):
                model = MultiscreenForCausalLM(config)
                self.assertTrue(
                    torch.equal(model.multiscreen.s_E, torch.zeros_like(model.multiscreen.s_E))
                )
                expected_s_f = torch.full_like(
                    model.multiscreen.s_F,
                    math.log(math.sqrt(config.hidden_size)),
                )
                self.assertTrue(torch.equal(model.multiscreen.s_F, expected_s_f))

                expected_s_o_value = math.log(
                    1.0 / math.sqrt(config.num_attention_heads * config.num_hidden_layers)
                )
                for layer in model.multiscreen.layers:
                    block = layer.block
                    expected_sw = torch.linspace(
                        0,
                        math.log(config.mipe_threshold),
                        config.num_attention_heads,
                        dtype=block.sw.dtype,
                        device=block.sw.device,
                    )
                    self.assertTrue(torch.equal(block.sw, expected_sw))
                    self.assertTrue(torch.equal(block.sr, torch.zeros_like(block.sr)))
                    self.assertTrue(
                        torch.equal(block.sO, torch.full_like(block.sO, expected_s_o_value))
                    )

    def test_fixed_seed_reproducibility_and_aggregate_statistical_sanity(self) -> None:
        config = MultiscreenConfig(
            vocab_size=1_024,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            key_dim=16,
            value_dim=64,
            initializer_range=0.1,
        )

        torch.manual_seed(20_260_806)
        first = MultiscreenForCausalLM(config)
        torch.manual_seed(20_260_806)
        second = MultiscreenForCausalLM(config)
        for key, first_tensor in first.state_dict().items():
            torch.testing.assert_close(second.state_dict()[key], first_tensor, rtol=0.0, atol=0.0)

        role_values: dict[str, list[torch.Tensor]] = {
            "embedding": [first.multiscreen.embed.weight],
            "qk": [],
            "value": [],
            "gate": [],
            "output": [],
        }
        for layer in first.multiscreen.layers:
            block = layer.block
            role_values["qk"].extend([block.q_proj.weight, block.k_proj.weight])
            role_values["value"].append(block.v_proj.weight)
            role_values["gate"].append(block.g_proj.weight)
            role_values["output"].append(block.o_proj.weight)

        role_stds = {
            "embedding": config.initializer_range / math.sqrt(config.hidden_size),
            "qk": config.initializer_range / math.sqrt(config.key_dim),
            "value": config.initializer_range / math.sqrt(config.value_dim),
            "gate": config.initializer_range,
            "output": config.initializer_range / math.sqrt(config.hidden_size),
        }
        for role, tensors in role_values.items():
            with self.subTest(role=role):
                standardized = torch.cat(
                    [tensor.detach().float().reshape(-1) / role_stds[role] for tensor in tensors]
                )
                self.assertGreaterEqual(standardized.numel(), 100_000)
                self.assertLess(abs(float(standardized.mean())), 0.02)
                population_std = float(standardized.std(unbiased=False))
                self.assertGreater(population_std, 0.95)
                self.assertLess(population_std, 1.05)


if __name__ == "__main__":
    unittest.main()
