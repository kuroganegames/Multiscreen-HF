import copy
import gc
import tempfile
import unittest
import weakref

import torch
import torch.nn.functional as F

from multiscreen_transformers import MultiscreenConfig, MultiscreenForCausalLM


def _model(vocab_size: int = 23) -> MultiscreenForCausalLM:
    return MultiscreenForCausalLM(
        MultiscreenConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            key_dim=4,
            value_dim=8,
        )
    )


class HFOutputHeadContractTest(unittest.TestCase):
    def test_output_head_is_callable_parameter_free_normalized_tie(self) -> None:
        model = _model()
        head = model.get_output_embeddings()
        self.assertIs(head, model.lm_head)
        self.assertIsNot(head, model.get_input_embeddings())
        self.assertEqual(list(head.parameters()), [])
        self.assertEqual(list(head.buffers()), [])
        self.assertFalse(any(key.startswith("lm_head.") for key in model.state_dict()))

        hidden = torch.randn(2, 4, model.config.hidden_size)
        expected_weight = F.normalize(model.multiscreen.embed.weight, dim=-1) * model.multiscreen.s_F.exp()
        self.assertEqual(head(hidden).shape, (2, 4, model.config.vocab_size))
        torch.testing.assert_close(head.weight, expected_weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(head(hidden), model._compute_logits(hidden), rtol=0.0, atol=0.0)

    def test_set_output_embeddings_is_atomic(self) -> None:
        model = _model()
        input_before = model.get_input_embeddings()
        head_before = model.lm_head
        state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}

        model.set_output_embeddings(head_before)
        self.assertIs(model.lm_head, head_before)
        with self.assertRaises(ValueError):
            model.set_output_embeddings(torch.nn.Linear(model.config.hidden_size, model.config.vocab_size))

        self.assertIs(model.get_input_embeddings(), input_before)
        self.assertIs(model.lm_head, head_before)
        self.assertEqual(state_before.keys(), model.state_dict().keys())
        for key, value in state_before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0.0, atol=0.0)

    def test_deepcopy_rebinds_owner_and_isolates_state_and_gradients(self) -> None:
        original = _model()
        original.eval()
        copied = copy.deepcopy(original)
        self.assertIs(copied.lm_head._owner_ref(), copied)
        self.assertIsNot(copied.lm_head._owner_ref(), original)
        self.assertFalse(copied.lm_head.training)
        self.assertEqual(list(copied.lm_head.children()), [])
        self.assertEqual(original.state_dict().keys(), copied.state_dict().keys())
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, copied.state_dict()[key], rtol=0.0, atol=0.0)

        hidden = torch.randn(2, 3, original.config.hidden_size)
        torch.testing.assert_close(original.lm_head(hidden), copied.lm_head(hidden), rtol=0.0, atol=0.0)
        original_logits = original.lm_head(hidden).detach().clone()
        with torch.no_grad():
            copied.multiscreen.embed.weight[0, 0].add_(1.0)
            copied.multiscreen.s_F.add_(0.25)
        self.assertFalse(torch.equal(original.lm_head(hidden), copied.lm_head(hidden)))
        torch.testing.assert_close(original.lm_head(hidden), original_logits, rtol=0.0, atol=0.0)
        copied_logits = copied.lm_head(hidden).detach().clone()
        with torch.no_grad():
            original.multiscreen.embed.weight[1, 1].sub_(0.75)
            original.multiscreen.s_F.sub_(0.1)
        torch.testing.assert_close(copied.lm_head(hidden), copied_logits, rtol=0.0, atol=0.0)

        original.zero_grad(set_to_none=True)
        copied.zero_grad(set_to_none=True)
        copied.lm_head(hidden).sum().backward()
        self.assertIsNone(original.multiscreen.embed.weight.grad)
        self.assertIsNone(original.multiscreen.s_F.grad)
        self.assertIsNotNone(copied.multiscreen.embed.weight.grad)
        self.assertIsNotNone(copied.multiscreen.s_F.grad)

        original_ref = weakref.ref(original)
        del original
        gc.collect()
        self.assertIsNone(original_ref())
        self.assertEqual(copied.lm_head(hidden).shape[-1], copied.config.vocab_size)

    def test_resize_expand_shrink_and_save_load(self) -> None:
        model = _model(23)
        original_keys = tuple(model.state_dict())
        original_prefix = model.multiscreen.embed.weight.detach().clone()
        original_non_embedding = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name != "multiscreen.embed.weight"
        )

        model.resize_token_embeddings(29, mean_resizing=False)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 29)
        self.assertEqual(model.config.vocab_size, 29)
        self.assertEqual(model.vocab_size, 29)
        self.assertEqual(model.lm_head.weight.shape, (29, model.config.hidden_size))
        torch.testing.assert_close(model.multiscreen.embed.weight[:23], original_prefix, rtol=0.0, atol=0.0)

        model.resize_token_embeddings(19, mean_resizing=False)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 19)
        self.assertEqual(model.config.vocab_size, 19)
        self.assertEqual(model.vocab_size, 19)
        self.assertEqual(tuple(model.state_dict()), original_keys)
        self.assertFalse(any(key.startswith("lm_head.") for key in model.state_dict()))
        self.assertEqual(
            sum(p.numel() for n, p in model.named_parameters() if n != "multiscreen.embed.weight"),
            original_non_embedding,
        )
        torch.testing.assert_close(model.multiscreen.embed.weight, original_prefix[:19], rtol=0.0, atol=0.0)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=False)
            loaded = MultiscreenForCausalLM.from_pretrained(directory)
        self.assertIs(loaded.get_output_embeddings(), loaded.lm_head)
        self.assertEqual(tuple(loaded.state_dict()), original_keys)
        hidden = torch.randn(1, 2, model.config.hidden_size)
        torch.testing.assert_close(model.lm_head(hidden), loaded.lm_head(hidden), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
