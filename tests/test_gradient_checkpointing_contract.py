"""Executable P1-preflight B gradient-checkpointing API contracts."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from multiscreen_transformers import (
    MultiscreenConfig,
    MultiscreenForCausalLM,
    MultiscreenPreTrainedModel,
)


LEGACY_WARNING = "old version of the checkpointing format"
MISSING_INPUT_GRAD_WARNING = "None of the inputs have requires_grad=True"


def make_config(*, gradient_checkpointing: bool = False) -> MultiscreenConfig:
    return MultiscreenConfig(
        vocab_size=43,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        key_dim=4,
        value_dim=8,
        max_position_embeddings=32,
        mipe_compute_dtype="fp32",
        softmask_compute_dtype="fp32",
        gradient_checkpointing=gradient_checkpointing,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def make_model(*, gradient_checkpointing: bool = False) -> MultiscreenForCausalLM:
    torch.manual_seed(20_260_809)
    return MultiscreenForCausalLM(
        make_config(gradient_checkpointing=gradient_checkpointing)
    )


def fixed_batch() -> tuple[torch.LongTensor, torch.LongTensor]:
    input_ids = torch.tensor(
        [[1, 7, 9, 4, 3, 8], [1, 5, 6, 2, 0, 0]],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[1, 4:] = -100
    return input_ids, labels


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def capture_checkpoint_diagnostics() -> Iterator[tuple[list[str], list[warnings.WarningMessage]]]:
    handler = _CollectingHandler()
    loggers = [
        logging.getLogger("transformers.modeling_utils"),
        logging.getLogger("torch.utils.checkpoint"),
    ]
    for target in loggers:
        target.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield handler.messages, caught
    finally:
        for target in loggers:
            target.removeHandler(handler)


def diagnostic_texts(
    log_messages: list[str],
    caught_warnings: list[warnings.WarningMessage],
) -> list[str]:
    return [*log_messages, *(str(item.message) for item in caught_warnings)]


def assert_no_target_warnings(
    testcase: unittest.TestCase,
    log_messages: list[str],
    caught_warnings: list[warnings.WarningMessage],
) -> None:
    messages = diagnostic_texts(log_messages, caught_warnings)
    testcase.assertFalse(
        any(LEGACY_WARNING in message for message in messages),
        messages,
    )
    testcase.assertFalse(
        any(MISSING_INPUT_GRAD_WARNING in message for message in messages),
        messages,
    )


class GradientCheckpointingApiContractTests(unittest.TestCase):
    def test_supported_hook_enable_disable_and_non_reentrant_default(self) -> None:
        self.assertTrue(MultiscreenPreTrainedModel.supports_gradient_checkpointing)
        self.assertNotIn(
            "_set_gradient_checkpointing",
            MultiscreenPreTrainedModel.__dict__,
        )

        model = make_model()
        signature = inspect.signature(model._set_gradient_checkpointing)
        self.assertNotIn("value", signature.parameters)
        self.assertFalse(model.is_gradient_checkpointing)
        self.assertFalse(model.multiscreen.gradient_checkpointing)

        caller_kwargs: dict[str, object] = {"preserve_rng_state": False}
        with capture_checkpoint_diagnostics() as (log_messages, caught):
            model.gradient_checkpointing_enable(caller_kwargs)
        self.assertEqual(
            caller_kwargs,
            {"preserve_rng_state": False},
            "the caller's kwargs mapping was mutated",
        )
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertTrue(model.multiscreen.gradient_checkpointing)

        installed = model.multiscreen._gradient_checkpointing_func
        self.assertIsInstance(installed, functools.partial)
        self.assertIs(installed.keywords.get("use_reentrant"), False)
        self.assertIs(installed.keywords.get("preserve_rng_state"), False)
        assert_no_target_warnings(self, log_messages, caught)

        calls: list[object] = []

        def injected(function, *args, **kwargs):
            calls.append(function)
            return function(*args, **kwargs)

        model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=injected,
        )
        with capture_checkpoint_diagnostics() as (log_messages, caught):
            model.gradient_checkpointing_disable()
        self.assertFalse(model.is_gradient_checkpointing)
        self.assertFalse(model.multiscreen.gradient_checkpointing)
        assert_no_target_warnings(self, log_messages, caught)

        input_ids, labels = fixed_batch()
        output = model.train()(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertEqual(calls, [], "disabled checkpoint function was invoked")

    def test_legacy_config_opt_in_installs_supported_function(self) -> None:
        with capture_checkpoint_diagnostics() as (log_messages, caught):
            model = make_model(gradient_checkpointing=True)

        self.assertTrue(model.is_gradient_checkpointing)
        self.assertTrue(model.multiscreen.gradient_checkpointing)
        installed = model.multiscreen._gradient_checkpointing_func
        self.assertIsInstance(installed, functools.partial)
        self.assertIs(installed.keywords.get("use_reentrant"), False)
        assert_no_target_warnings(self, log_messages, caught)

    def test_committed_tokenizer_loads_in_both_compatibility_lanes(self) -> None:
        from scripts.p0_3_tinystories_stability import load_tokenizer_compat

        tokenizer_path = Path(__file__).resolve().parents[1] / "tokenizers" / "tinystories_spm768"
        tokenizer = load_tokenizer_compat(tokenizer_path, cache_dir=None)
        self.assertEqual(len(tokenizer), 768)
        self.assertEqual(tokenizer.unk_token_id, 0)
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 3)
        self.assertEqual(tokenizer.model_input_names, ["input_ids", "attention_mask"])
        encoded = tokenizer("Once upon a time", return_tensors="pt", add_special_tokens=False)
        self.assertEqual(set(encoded), {"input_ids", "attention_mask"})

    def test_installed_and_custom_checkpoint_functions_are_invoked(self) -> None:
        model = make_model().train()
        model.gradient_checkpointing_enable()
        installed = model.multiscreen._gradient_checkpointing_func
        calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def injected(function, *args, **kwargs):
            calls.append((function, args, kwargs))
            return installed(function, *args, **kwargs)

        model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=injected,
        )
        input_ids, labels = fixed_batch()
        with capture_checkpoint_diagnostics() as (log_messages, caught):
            output = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            self.assertIsNotNone(output.loss)
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()

        self.assertEqual(len(calls), model.config.num_hidden_layers)
        self.assertTrue(all(callable(function) for function, _, _ in calls))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        assert_no_target_warnings(self, log_messages, caught)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.multiscreen.embed.weight.detach().clone()
        optimizer.step()
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        self.assertFalse(torch.equal(before, model.multiscreen.embed.weight.detach()))

    def test_non_reentrant_path_handles_a_no_grad_checkpoint_input(self) -> None:
        model = make_model().train()
        model.multiscreen.embed.weight.requires_grad_(False)
        model.multiscreen.s_E.requires_grad_(False)
        model.gradient_checkpointing_enable()
        installed = model.multiscreen._gradient_checkpointing_func
        checkpoint_input_requires_grad: list[bool] = []

        def observing_checkpoint(function, *args, **kwargs):
            checkpoint_input_requires_grad.append(bool(args[0].requires_grad))
            return installed(function, *args, **kwargs)

        model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=observing_checkpoint,
        )
        input_ids, labels = fixed_batch()
        with capture_checkpoint_diagnostics() as (log_messages, caught):
            output = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            self.assertIsNotNone(output.loss)
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()

        self.assertEqual(len(checkpoint_input_requires_grad), model.config.num_hidden_layers)
        self.assertFalse(
            checkpoint_input_requires_grad[0],
            "the first checkpoint input unexpectedly required gradients",
        )
        layer_parameters = list(model.multiscreen.layers.parameters())
        self.assertTrue(layer_parameters)
        self.assertTrue(all(parameter.grad is not None for parameter in layer_parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in layer_parameters))
        assert_no_target_warnings(self, log_messages, caught)

    def test_checkpointed_and_plain_forward_loss_and_gradients_match(self) -> None:
        plain = make_model().train()
        checkpointed = make_model().train()
        checkpointed.load_state_dict(plain.state_dict())
        checkpointed.gradient_checkpointing_enable()
        input_ids, labels = fixed_batch()

        plain_output = plain(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        checkpointed_output = checkpointed(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        self.assertIsNotNone(plain_output.loss)
        self.assertIsNotNone(checkpointed_output.loss)
        plain_output.loss.backward()
        checkpointed_output.loss.backward()

        torch.testing.assert_close(
            checkpointed_output.logits,
            plain_output.logits,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            checkpointed_output.loss,
            plain_output.loss,
            rtol=1e-6,
            atol=1e-7,
        )
        plain_parameters = dict(plain.named_parameters())
        checkpointed_parameters = dict(checkpointed.named_parameters())
        self.assertEqual(plain_parameters.keys(), checkpointed_parameters.keys())
        for name, plain_parameter in plain_parameters.items():
            with self.subTest(parameter=name):
                checkpointed_parameter = checkpointed_parameters[name]
                self.assertIsNotNone(plain_parameter.grad)
                self.assertIsNotNone(checkpointed_parameter.grad)
                torch.testing.assert_close(
                    checkpointed_parameter.grad,
                    plain_parameter.grad,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_transient_function_is_not_serialized_and_reload_generates(self) -> None:
        model = make_model().eval()
        calls: list[object] = []

        def custom_checkpoint(function, *args, **kwargs):
            calls.append(function)
            return function(*args, **kwargs)

        model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=custom_checkpoint,
        )
        self.assertFalse(any("gradient_checkpointing_func" in key for key in model.state_dict()))
        config_json = json.dumps(model.config.to_dict(), sort_keys=True)
        self.assertNotIn("gradient_checkpointing_func", config_json)
        self.assertNotIn("use_reentrant", config_json)

        input_ids, _ = fixed_batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model.save_pretrained(path, safe_serialization=True)
            saved_config = (path / "config.json").read_text(encoding="utf-8")
            self.assertNotIn("gradient_checkpointing_func", saved_config)
            self.assertNotIn("use_reentrant", saved_config)

            loaded = MultiscreenForCausalLM.from_pretrained(path).eval()
            self.assertFalse(loaded.is_gradient_checkpointing)
            self.assertFalse(hasattr(loaded.multiscreen, "_gradient_checkpointing_func"))
            with torch.no_grad():
                expected = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                actual = loaded(input_ids=input_ids, use_cache=False, return_dict=True).logits
                generation_kwargs = {
                    "input_ids": input_ids[:1, :3],
                    "max_new_tokens": 2,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": 0,
                    "eos_token_id": None,
                }
                expected_generated = model.generate(**generation_kwargs)
                actual_generated = loaded.generate(
                    input_ids=input_ids[:1, :3],
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=0,
                    eos_token_id=None,
                )
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual_generated, expected_generated, rtol=0.0, atol=0.0)
            self.assertEqual(tuple(actual_generated.shape), (1, 5))


if __name__ == "__main__":
    unittest.main()
