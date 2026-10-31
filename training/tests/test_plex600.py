"""Small-configuration checks for the separate Plex 600M architecture.

Never instantiate the 600M default model in CI; parameter count is analytic.
"""

import unittest

import torch

from plex_training.plex600 import Plex600Config, Plex600Model, parameter_count


def small_config(**overrides) -> Plex600Config:
    fields = dict(
        vocab_size=128, context_length=32, width=64, layers=2,
        attention_heads=4, key_value_heads=2, feed_forward_width=128,
        dropout=0.0,
    )
    fields.update(overrides)
    return Plex600Config(**fields)


class Plex600ArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = small_config()
        self.model = Plex600Model(self.config)

    def test_default_parameter_count_without_allocating_600m(self) -> None:
        self.assertEqual(Plex600Config().parameter_count(), 600_372_480)
        self.assertEqual(parameter_count(self.model), self.config.parameter_count())
        self.assertEqual(self.model.config.to_dict()["key_value_heads"], 2)

    def test_gqa_and_tied_output_projection(self) -> None:
        attention = self.model.blocks[0].attention
        self.assertEqual(attention.q_proj.weight.shape, (64, 64))
        self.assertEqual(attention.k_proj.weight.shape, (32, 64))
        self.assertEqual(attention.v_proj.weight.shape, (32, 64))
        self.assertFalse(hasattr(self.model, "lm_head"))
        self.assertFalse(hasattr(self.model, "position_embeddings"))

    def test_training_forward_backward_and_ignored_targets(self) -> None:
        inputs = torch.randint(0, self.config.vocab_size, (2, 9))
        targets = inputs.roll(shifts=-1, dims=1)
        targets[:, -1] = -100
        logits, loss = self.model(inputs, targets)
        self.assertEqual(tuple(logits.shape), (2, 9, 128))
        self.assertIsNotNone(loss)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters()))

    def test_causal_mask_prevents_future_token_leaks(self) -> None:
        self.model.eval()
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 14, 15, 16]])
        with torch.no_grad():
            original_logits, _ = self.model(inputs)
            changed_logits, _ = self.model(changed)
        self.assertTrue(torch.allclose(original_logits[:, :3], changed_logits[:, :3], atol=1e-5))

    def test_weighted_loss_ignores_masked_positions(self) -> None:
        self.model.eval()
        inputs = torch.randint(0, 128, (1, 8))
        targets = inputs.roll(shifts=-1, dims=1)
        targets[:, -1] = -100
        weights = torch.tensor([[1., 2., 2., 0., 0., 0., 0., 0.]])
        changed_targets = targets.clone()
        changed_targets[:, 3:7] = (changed_targets[:, 3:7] + 1) % 128
        _, loss_a = self.model(inputs, targets, target_weights=weights)
        _, loss_b = self.model(inputs, changed_targets, target_weights=weights)
        self.assertTrue(torch.allclose(loss_a, loss_b, atol=1e-6))

    def test_checkpointing_backward_on_small_model(self) -> None:
        self.model.gradient_checkpointing_enable()
        inputs = torch.randint(0, 128, (1, 6))
        targets = inputs.roll(shifts=-1, dims=1)
        _, loss = self.model(inputs, targets)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in self.model.parameters()))
        self.model.gradient_checkpointing_enable(False)
        self.assertFalse(self.model.gradient_checkpointing)

    def test_input_target_and_config_validation(self) -> None:
        inputs = torch.randint(0, 128, (1, 5))
        with self.assertRaises(ValueError):
            self.model(inputs.float())
        with self.assertRaises(ValueError):
            self.model(inputs, torch.full_like(inputs, -100))
        with self.assertRaises(ValueError):
            self.model(inputs, inputs, target_weights=torch.zeros_like(inputs).float())
        with self.assertRaises(ValueError):
            self.model(torch.full_like(inputs, 128))
        for variant in [
            {"width": 65}, {"attention_heads": 3}, {"key_value_heads": 3},
            {"context_length": 0}, {"dropout": float("nan")}, {"norm_epsilon": -1.0},
        ]:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                small_config(**variant)


if __name__ == "__main__":
    unittest.main()
