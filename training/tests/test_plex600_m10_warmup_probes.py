"""Small CPU controls for checkpoint activation exchanges; no optimizer."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from plex_training import plex600_m10_warmup_probes as probe


class WarmupProbeTests(unittest.TestCase):
    def test_plan_never_opens_inputs(self):
        with patch.object(probe, "run", side_effect=AssertionError), redirect_stdout(StringIO()) as stream:
            probe.main([])
        self.assertFalse(json.loads(stream.getvalue())["modelLoaded"])

    def test_different_report_refused_before_model_load(self):
        with TemporaryDirectory() as temp:
            (Path(temp) / "report.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "November 18"):
                probe.main(["--run-inference", "--run-dir", temp, "--m9-backup", temp,
                            "--baseline", temp, "--bundle", temp])

    def test_only_boundary_token_is_replaced(self):
        prompt, answer = [8, 6, 4], [7, 4]
        contexts = probe.contexts(prompt, answer, newline=4, space=9)
        self.assertEqual(contexts["promptSpace"], [8, 6, 9])
        self.assertEqual(contexts["goldSpace"], [8, 6, 4, 7, 9])
        self.assertEqual(contexts["goldBeforeNewline"], [8, 6, 4, 7])
        self.assertEqual((prompt, answer), ([8, 6, 4], [7, 4]))
        with self.assertRaisesRegex(ValueError, "newline"):
            probe.contexts(prompt, [7, 6], newline=4, space=9)

    def test_native_parity_and_final_position_exchange(self):
        import torch
        from plex_training.plex600 import Plex600Config, Plex600Model

        config = Plex600Config(vocab_size=32, width=16, layers=2, attention_heads=2,
                               key_value_heads=1, feed_forward_width=32)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(32)
            models = {key: Plex600Model(config).eval() for key in ("M9", "M10-10")}
            inputs = torch.tensor([[8, 10, 4]])
            rows = probe.sweep(torch, models, inputs, 7)
            for receiver, donor in (("M9", "M10-10"), ("M10-10", "M9")):
                indexed = {(row["cut"], row["positions"]): row for row in rows
                           if row["receiver"] == receiver and row["donor"] == donor}
                self.assertEqual(len(indexed), 9)
                native, _ = models[receiver](inputs)
                baseline = probe.metrics(native, 7)
                for key, value in baseline.items():
                    self.assertEqual(indexed[2, "preceding"][key], value)
                    self.assertEqual(indexed[2, "last"][key], indexed[2, "all"][key])
                # At cut zero only the input embeddings come from the donor.
                model = models[receiver]
                hidden = models[donor].token_embedding(inputs)
                for block in model.blocks:
                    hidden = block(hidden, *model.rotary(3, inputs.device))
                expected = torch.nn.functional.linear(model.final_norm(hidden),
                                                      model.token_embedding.weight)
                self.assertEqual(indexed[0, "all"]["eosMinusFirstLogit"],
                                 probe.metrics(expected, 7)["eosMinusFirstLogit"])

    def test_source_pair_changes_only_the_supplied_code(self):
        from types import SimpleNamespace
        tokenizer = SimpleNamespace(encode=lambda text: list(map(ord, text)))
        for case in probe.copy.rows():
            inputs, tokens, metadata = probe.source_pair(tokenizer, case)
            index = probe.one_difference(*inputs)
            self.assertEqual(index, metadata["sourcePosition"])
            self.assertEqual(tokens, [ids[index] for ids in inputs])
            self.assertLess(index, len(inputs[0]) - metadata["answerPosition"])
        with self.assertRaisesRegex(ValueError, "one changed token"):
            probe.one_difference([5, 6], [5])

    def test_gold_prefixes_score_the_shifted_content_and_eos(self):
        import torch
        logits = torch.zeros((1, 5, 8))
        logits[0, 2, 5] = logits[0, 3, 4] = logits[0, 4, 3] = 2.0
        measured = probe.gold_prefix_scores(torch, logits, 3, [5, 4])
        self.assertEqual(measured["targetIds"], measured["greedyIds"])
        self.assertEqual(measured["eosMinusTargetLogits"], [-2.0, -2.0, 0.0])

    def test_source_scorer_detects_copying_and_constant_choice(self):
        import torch
        from types import SimpleNamespace
        case = next(r for r in probe.copy.rows() if r["id"] == "m10-js-counter")
        tokenizer = SimpleNamespace(encode=lambda text: list(map(ord, text)))
        _, pair, metadata = probe.source_pair(tokenizer, case)

        class CopyOrConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
                self.constant = False

            def forward(self, ids):
                selected = pair[0] if self.constant else int(ids[0, metadata["sourcePosition"]])
                output = self.anchor.new_zeros(1, 1, 256)
                output[0, 0, selected] = 2
                return output, None

        model = CopyOrConstant()
        copied = probe.probe_source(torch, model, tokenizer, case)
        self.assertTrue(copied["prefersMatchingTokenBothWays"])
        self.assertEqual(copied["pairedChange"], 4)
        model.constant = True
        constant = probe.probe_source(torch, model, tokenizer, case)
        self.assertFalse(constant["prefersMatchingTokenBothWays"])
        self.assertEqual(constant["pairedChange"], 0)

    def test_seed_is_counted_and_only_first_decision_is_forced(self):
        import torch
        from types import SimpleNamespace
        tokenizer = SimpleNamespace(vocabulary_size=8, decode=lambda ids: {5: "a", 6: "b"}[ids[0]])
        logits = torch.tensor([0., 0., 0., 9., 0., 8., 1., 0.])
        with patch.object(probe.m8, "generate_one", return_value=("z", 1, "eos")) as generate:
            rows = probe.continuations(torch, None, tokenizer, [7], [6], logits, {})
        self.assertEqual([(r["output"], r["correctFirst"], r["generatedTokensIncludingSeed"])
                          for r in rows], [("az", False, 2), ("bz", True, 2)])
        for call, seed in zip(generate.call_args_list, (5, 6)):
            self.assertEqual(call.args[3], [7, seed])
            self.assertEqual(call.kwargs["max_new_tokens"], probe.copy.MAX_GENERATION - 1)
            self.assertEqual(call.kwargs["decoding_policy"], "fitted-vocab-v1")
