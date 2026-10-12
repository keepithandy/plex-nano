from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "training" / "tests"
REQUIRED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "training-tests.yml"
EXPERIMENT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "training-experiment-tests.yml"
WINDOWS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "training-windows-tests.yml"
TEST_MODULE_PATTERN = re.compile(r"training/tests/(test_[A-Za-z0-9_]+\.py)")


def _workflow_modules(path: Path) -> list[str]:
    return TEST_MODULE_PATTERN.findall(path.read_text(encoding="utf-8"))


class CiTestInventoryTests(unittest.TestCase):
    def test_every_top_level_test_module_is_classified_exactly_once(self) -> None:
        discovered = {
            path.name
            for path in TEST_ROOT.iterdir()
            if path.is_file() and path.name.startswith("test_") and path.suffix == ".py"
        }
        required = _workflow_modules(REQUIRED_WORKFLOW)
        experiment = _workflow_modules(EXPERIMENT_WORKFLOW)

        self.assertEqual(len(required), len(set(required)), "required suite contains duplicate modules")
        self.assertEqual(len(experiment), len(set(experiment)), "experiment suite contains duplicate modules")
        self.assertEqual(set(required) & set(experiment), set(), "test module appears in both suites")
        self.assertEqual(
            discovered,
            set(required) | set(experiment),
            "every top-level test module must be assigned to exactly one CI suite",
        )

    def test_required_suite_remains_automatic(self) -> None:
        trigger_block = REQUIRED_WORKFLOW.read_text(encoding="utf-8").split("permissions:", 1)[0]
        self.assertIn("pull_request:", trigger_block)
        self.assertIn("push:", trigger_block)
        self.assertIn("- master", trigger_block)
        self.assertIn("workflow_dispatch:", trigger_block)

    def test_root_line_ending_policy_triggers_linux_and_windows_training_ci(self) -> None:
        for workflow in (REQUIRED_WORKFLOW, WINDOWS_WORKFLOW):
            with self.subTest(workflow=workflow.name):
                trigger_block = workflow.read_text(encoding="utf-8").split("permissions:", 1)[0]
                self.assertEqual(
                    trigger_block.count('- ".gitattributes"'),
                    2,
                    ".gitattributes must trigger both pull_request and master push verification",
                )

    def test_experiment_suite_remains_manual_only(self) -> None:
        trigger_block = EXPERIMENT_WORKFLOW.read_text(encoding="utf-8").split("permissions:", 1)[0]
        self.assertIn("workflow_dispatch:", trigger_block)
        self.assertNotIn("pull_request:", trigger_block)
        self.assertNotIn("\n  push:", trigger_block)


if __name__ == "__main__":
    unittest.main()
