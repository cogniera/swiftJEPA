#Author : Paarth Sharma
#File Name : test_benchmark.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Check raw sample reporting and CUDA timing of host preprocessing.

import csv
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from benchmarks.benchmark_vjepa2 import main, make_events, measure_iteration, parse_args, save_outputs


# Persist enough data to reconstruct statistics and reject an unvalidated quality claim.
class OutputTests(unittest.TestCase):
    def test_default_cuda_device_has_an_explicit_index(self):
        self.assertEqual(parse_args([]).device, "cuda:0")
        self.assertEqual(parse_args(["--device", "cuda"]).device, "cuda:0")
        self.assertEqual(parse_args(["--device", "cuda:2"]).device, "cuda:2")

    def test_raw_samples_and_median_survive_serialization(self):
        samples = [{"end_to_end_ms": value, "wall_ms": value + 0.1,
                    "preprocess_cpu_ms": value - 1, "h2d_copy_ms": 0.2,
                    "encoder_cuda_ms": 0.8, "gpu_total_ms": 1.0}
                   for value in [10.0] * 180 + [12.0] * 20]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            summary = save_outputs(parse_args([]),
                                   {"samples": samples, "instrumentation": {}},
                                   Path(directory), {"tracked_dirty": False})
            saved = json.loads((Path(directory) / "summary.json").read_text())
            with (Path(directory) / "timings_per_iteration.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 200)
        self.assertEqual(float(rows[-1]["end_to_end_ms"]), 12.0)
        self.assertEqual(saved, summary)
        self.assertEqual(saved["stages"]["end_to_end_ms"]["median_ms"], 10.0)
        self.assertTrue(saved["validity"]["passed"])
        self.assertFalse(saved["acceptance_eligible"])
        self.assertIsNone(saved["accuracy"]["passed"])

    def test_compile_activity_and_eager_fallback_are_rejected(self):
        samples = [{"end_to_end_ms": 10.0}] * 200
        for before, after, expected in (
            ({"stats": {"unique_graphs": 1}}, {"stats": {"unique_graphs": 2}}, "compiler activity"),
            ({"stats": {}}, {"stats": {}}, "captured no graphs"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                results = {"samples": samples, "instrumentation": {},
                           "compilation": {"enabled": True,
                                           "before_measured_iterations": before,
                                           "after_measured_iterations": after}}
                summary = save_outputs(parse_args(["--compile"]), results, Path(directory),
                                       {"tracked_dirty": False}, {"passed": True})
            self.assertFalse(summary["validity"]["passed"])
            self.assertFalse(summary["acceptance_eligible"])
            self.assertTrue(any(expected in reason for reason in summary["validity"]["reasons"]))

    def test_quality_failure_cannot_be_an_accepted_latency_claim(self):
        results = {"samples": [{"end_to_end_ms": 10.0}] * 200, "instrumentation": {}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            summary = save_outputs(parse_args([]), results, Path(directory),
                                   {"tracked_dirty": False}, {"passed": False})
        self.assertTrue(summary["validity"]["passed"])
        self.assertFalse(summary["acceptance_eligible"])


# Required official comparators preserve failed quality reports while still producing latency data.
class OfficialBaselinePolicyTests(unittest.TestCase):
    def test_failed_eager_and_compiled_quality_still_produces_latency(self):
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
        accuracy = {
            "passed": False, "minimum_cosine": 0.997714, "minimum_cosine_required": 0.999,
            "clips": [
                {"name": "moving_color_bars", "passed": False, "minimum_cosine": 0.997714},
                {"name": "midgray", "passed": False, "minimum_cosine": 0.998788},
            ],
        }
        samples = [{"end_to_end_ms": 10.0, "wall_ms": 10.1, "preprocess_cpu_ms": 5.0,
                    "h2d_copy_ms": 0.2, "encoder_cuda_ms": 4.8, "gpu_total_ms": 5.0}] * 200
        for compiled in (False, True):
            with self.subTest(compiled=compiled), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                out_dir = Path(directory) / "run"
                argv = ["--out", str(out_dir), "--session-id", "policy-test"]
                if compiled:
                    argv.append("--compile")
                args = parse_args(argv)
                encoder = torch.nn.Identity()
                counters = {"stats": {"unique_graphs": 1}}
                results = {"samples": samples, "instrumentation": {}, "compilation": {
                    "enabled": compiled, "before_measured_iterations": counters,
                    "after_measured_iterations": counters,
                }}
                output = io.StringIO()

                # Replace unavailable GPU work while exercising real artifact writing and main control flow.
                with (
                    patch("benchmarks.benchmark_vjepa2.parse_args", return_value=args),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.set_device"),
                    patch("torch.cuda.device", return_value=nullcontext()),
                    patch("torch.set_num_threads"),
                    patch("cv2.setNumThreads"),
                    patch("benchmarks.benchmark_vjepa2.collect_environment", return_value={"tracked_dirty": False}),
                    patch("benchmarks.benchmark_vjepa2.build_encoder", return_value=(encoder, 256)),
                    patch("benchmarks.benchmark_vjepa2.make_preprocess", return_value=lambda clip: clip),
                    patch("torch.compile", return_value=encoder) as compiler,
                    patch("benchmarks.accuracy.validate_encoder", return_value=accuracy),
                    patch("benchmarks.benchmark_vjepa2.run_benchmark", return_value=results) as measure,
                    redirect_stdout(output),
                ):
                    main()
                measure.assert_called_once_with(args, encoder, 256, out_dir)
                if compiled:
                    compiler.assert_called_once_with(encoder, mode="max-autotune")
                else:
                    compiler.assert_not_called()
                saved_accuracy = json.loads((out_dir / "accuracy.json").read_text())
                summary = json.loads((out_dir / "summary.json").read_text())
                self.assertEqual(saved_accuracy, accuracy)
                self.assertEqual(summary["accuracy"], accuracy)
                self.assertTrue(summary["validity"]["passed"])
                self.assertFalse(summary["acceptance_eligible"])
                self.assertEqual(summary["stages"]["end_to_end_ms"]["count"], 200)
                self.assertTrue((out_dir / "timings_per_iteration.csv").is_file())
                self.assertIn("official baseline quality failed", output.getvalue())


# Real CUDA events must include host work between the synchronized start and upload.
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA events")
class CudaTimingTests(unittest.TestCase):
    def test_cpu_preprocessing_delay_is_in_end_to_end_metric(self):
        device = torch.device(parse_args([]).device)

        def preprocess(_):
            time.sleep(0.02)
            return torch.ones(1, 3, 2, 4, 4)

        with torch.cuda.device(device), torch.inference_mode():
            torch.cuda.set_device(device)
            stream = torch.cuda.current_stream(device)
            events = make_events(stream)
            row = measure_iteration(lambda x: x + 1, preprocess, None, device,
                                    torch.float32, events, stream)
        self.assertGreaterEqual(row["preprocess_cpu_ms"], 19)
        self.assertGreaterEqual(row["end_to_end_ms"], row["preprocess_cpu_ms"])
        self.assertGreater(row["end_to_end_ms"], row["gpu_total_ms"] + 18)
        self.assertGreaterEqual(row["wall_ms"] + 2, row["end_to_end_ms"])


if __name__ == "__main__":
    unittest.main()
