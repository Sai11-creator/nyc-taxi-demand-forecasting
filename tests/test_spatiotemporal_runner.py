"""Check artifact compatibility and orchestration without training a model."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatiotemporal_forecasting as runner


class LastHour(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, -1, :]


class RunnerCompatibilityTests(unittest.TestCase):
    def test_report_artifact_names(self):
        inputs = np.zeros((2, 24, 263), dtype=np.float32)
        targets = np.zeros((2, 263), dtype=np.float32)
        cases = [
            (
                "development_oct_dec_2024",
                "development_oct_dec_2024_predictions.npz",
                "development_test_metrics.json",
            ),
            (
                "original_2025",
                "original_2025_jan02_dec31_predictions.npz",
                "original_2025_metrics.json",
            ),
            ("full_2025", "full_year_2025_predictions.npz", "full_2025_metrics.json"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for name, predictions, metrics in cases:
                result = runner.evaluate(
                    LastHour(),
                    runner.loader(inputs, targets, False),
                    torch.device("cpu"),
                    0,
                    1,
                    output,
                    name,
                )
                self.assertEqual(result["rmse"], 0)
                self.assertTrue((output / metrics).is_file())
                with np.load(output / predictions) as saved:
                    np.testing.assert_array_equal(saved["prediction"], targets)

    def run_pipeline(self, output, fail_stage=None):
        calls = []

        def fake_run(command, check):
            self.assertTrue(check)
            self.assertEqual(Path(command[1]).name, "spatiotemporal_forecasting.py")
            stage = command[command.index("--stage") + 1]
            calls.append(command)
            if stage == fail_stage:
                raise subprocess.CalledProcessError(1, command)
            if stage == "develop":
                runner.save_json(
                    output, "development_selection.json", {"best_epoch": 3}
                )
            runner.save_json(output, f"{stage}_environment.json", {"device": "cpu"})

        arguments = [
            "spatiotemporal_forecasting.py",
            "--stage",
            "all",
            "--output-dir",
            str(output),
        ]
        with patch.object(
            runner.platform, "platform", return_value="test-platform"
        ), patch.object(sys, "argv", arguments), patch.object(
            runner.subprocess, "run", side_effect=fake_run
        ):
            runner.main()
        return calls

    def test_pipeline_selects_refit_duration_and_records_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            calls = self.run_pipeline(output)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1][calls[1].index("--epochs") + 1], "3")
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["refit_epochs_selected_on_validation"], 3)

    def test_failed_stage_does_not_record_success(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_pipeline(output, fail_stage="refit")
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_stage"], "refit")


if __name__ == "__main__":
    unittest.main()
