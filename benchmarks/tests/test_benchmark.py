#Author : Paarth Sharma
#File Name : test_benchmark.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Check raw sample reporting and CUDA timing of host preprocessing.

import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

from benchmarks.benchmark_vjepa2 import make_events, measure_iteration, parse_args, save_outputs


# Persist enough data to reconstruct statistics and reject an unvalidated quality claim.
class OutputTests(unittest.TestCase):
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


# Real CUDA events must include host work between the synchronized start and upload.
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA events")
class CudaTimingTests(unittest.TestCase):
    def test_cpu_preprocessing_delay_is_in_end_to_end_metric(self):
        device = torch.device("cuda", 0)

        def preprocess(_):
            time.sleep(0.02)
            return torch.ones(1, 3, 2, 4, 4)

        with torch.cuda.device(device), torch.inference_mode():
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
