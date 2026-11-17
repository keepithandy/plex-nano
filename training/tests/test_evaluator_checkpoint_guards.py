"""Regression for actual evaluator checkpoint guards; no GPU or training."""
import ast
from collections import OrderedDict
from pathlib import Path
import unittest

class CheckpointGuardTests(unittest.TestCase):
    def test_evaluators_accept_native_state_dict_and_reject_other_payloads(self):
        root=Path(__file__).resolve().parents[1]/"src/plex_training"
        for name in ("plex600_m8_eval.py","plex600_m8_teacher_forced.py"):
            tree=ast.parse((root/name).read_text(encoding="utf8"))
            guards=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test).startswith("type(state)")]
            self.assertEqual(len(guards),1)
            test=compile(ast.Expression(guards[0].test),name,"eval")
            for state,rejected in (({},False),(OrderedDict(),False),([],True),(None,True)):
                with self.subTest(module=name,state=type(state)):
                    self.assertEqual(eval(test,{"state":state,"OrderedDict":OrderedDict}),rejected)
