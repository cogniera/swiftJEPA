#Author : Paarth Sharma
#File Name : test_reporting.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Tests for benchmark statistics validity gates and optional metadata failures

"""Run with python -m unittest discover -s benchmarks/tests -p test_reporting.py."""

# Keep all reporting tests on the CPU with controlled metadata sources.
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.reporting import collect_environment, run_validity, summarize_samples


# Hand-calculated statistics distinguish median from mean and define percentile interpolation.
class StatisticsTests(unittest.TestCase):
    def test_summary_reports_mean_median_and_p95(self):
        summary = summarize_samples([1.0, 2.0, 4.0, 9.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean_ms"], 4.0)
        self.assertEqual(summary["median_ms"], 3.0)
        self.assertAlmostEqual(summary["p95_ms"], 8.25)
        json.dumps(summary, allow_nan=False)

    def test_summary_accepts_one_positive_sample(self):
        self.assertEqual(
            summarize_samples([2.5]),
            {"count": 1, "mean_ms": 2.5, "median_ms": 2.5, "p95_ms": 2.5},
        )

    def test_summary_rejects_invalid_samples(self):
        for samples in ([], [[1, 2]], [0], [-1], [float("nan")], [float("inf")], [float("-inf")], ["invalid"]):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    summarize_samples(samples)

    def test_large_finite_samples_remain_json_safe(self):
        summary = summarize_samples([1e308] * 200)
        for name in ("mean_ms", "median_ms", "p95_ms"):
            self.assertEqual(summary[name], 1e308)
        json.dumps(summary, allow_nan=False)
        validity = run_validity([1e308] * 200)
        self.assertTrue(validity["passed"])
        self.assertEqual(validity["p95_mean_ratio"], 1.0)
        json.dumps(validity, allow_nan=False)


# Exercise every independent acceptance condition and reject unusable timing values.
class ValidityTests(unittest.TestCase):
    def test_stable_clean_run_with_minimum_samples_passes(self):
        validity = run_validity([10.0] * 200)
        self.assertTrue(validity["passed"])
        self.assertEqual(validity["reasons"], [])
        self.assertEqual(validity["sample_count"], 200)
        self.assertEqual(validity["p95_mean_ratio"], 1.0)
        json.dumps(validity, allow_nan=False)

    def test_short_run_is_rejected_even_with_stable_samples(self):
        validity = run_validity([10.0] * 199)
        self.assertFalse(validity["passed"])
        self.assertEqual(validity["p95_mean_ratio"], 1.0)
        self.assertEqual(validity["reasons"], ["requires at least 200 measured iterations"])

    def test_unstable_tail_is_rejected_with_enough_samples(self):
        validity = run_validity([10.0] * 180 + [30.0] * 20)
        self.assertFalse(validity["passed"])
        self.assertEqual(validity["sample_count"], 200)
        self.assertAlmostEqual(validity["p95_mean_ratio"], 2.5)
        self.assertEqual(validity["reasons"], ["p95 to mean ratio exceeds 1.3"])

    def test_ratio_equal_to_limit_passes(self):
        samples = [10.0] * 180 + [30.0] * 20
        self.assertTrue(run_validity(samples, maximum_p95_mean=2.5)["passed"])

    def test_dirty_or_unknown_source_is_rejected(self):
        for source_dirty in (True, None):
            with self.subTest(source_dirty=source_dirty):
                validity = run_validity([10.0] * 200, source_dirty=source_dirty)
                self.assertFalse(validity["passed"])
                self.assertEqual(len(validity["reasons"]), 1)
                self.assertIn("tracked source", validity["reasons"][0])

    def test_rejection_reports_all_independent_causes(self):
        validity = run_validity([10.0] * 90 + [30.0] * 10, source_dirty=True)
        self.assertFalse(validity["passed"])
        self.assertEqual(len(validity["reasons"]), 3)

    def test_nonfinite_and_nonpositive_samples_are_json_safe_rejections(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                validity = run_validity([10.0] * 199 + [invalid])
                self.assertFalse(validity["passed"])
                self.assertIsNone(validity["p95_mean_ratio"])
                self.assertIn("samples must be finite and strictly positive", validity["reasons"])
                json.dumps(validity, allow_nan=False)

    def test_invalid_gate_configuration_is_rejected(self):
        for count in (0, -1):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    run_validity([1], minimum_iters=count)
        for limit in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    run_validity([1], maximum_p95_mean=limit)


# Optional tools must not crash metadata collection or imply unknown source is clean.
class EnvironmentTests(unittest.TestCase):
    @staticmethod
    def fake_torch():
        return SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda=None),
            get_num_threads=lambda: 4,
            get_num_interop_threads=lambda: 2,
            get_float32_matmul_precision=lambda: "highest",
            device=lambda _: SimpleNamespace(type="cpu", index=None),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=True, benchmark=False, deterministic=True),
            ),
        )

    def test_missing_commands_are_recorded_without_failure(self):
        fake_cv2 = SimpleNamespace(__version__="test", getNumThreads=lambda: 1)
        with patch.dict(sys.modules, {"torch": self.fake_torch(), "cv2": fake_cv2}), patch(
            "benchmarks.reporting._package_versions", return_value={"torch": "test"}
        ), patch("benchmarks.reporting.subprocess.run", side_effect=FileNotFoundError):
            environment = collect_environment(Path.cwd(), "cpu")

        self.assertIsNone(environment["commit"])
        self.assertIsNone(environment["tracked_dirty"])
        self.assertFalse(environment["nvidia_smi"]["available"])
        self.assertFalse(environment["nvcc"]["available"])
        self.assertIsNone(environment["gpu"]["uuid"])
        self.assertIsInstance(environment["hostname"], str)
        self.assertEqual(environment["cpu_threads"], {"torch": 4, "torch_interop": 2, "opencv": 1})
        self.assertEqual(environment["tf32"], {"matmul": False, "cudnn": True})
        json.dumps(environment, allow_nan=False)

    def test_tracked_dirty_source_and_full_command_output_are_captured(self):
        revision = "a" * 40
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[:2] == ["git", "rev-parse"]:
                output = revision + "\n"
            elif command[:2] == ["git", "status"]:
                output = " M benchmarks/benchmark_vjepa2.py\n"
            else:
                output = "first line\nsecond line\n"
            return subprocess.CompletedProcess(command, 0, stdout=output)

        fake_cv2 = SimpleNamespace(__version__="test", getNumThreads=lambda: 1)
        with patch.dict(sys.modules, {"torch": self.fake_torch(), "cv2": fake_cv2}), patch(
            "benchmarks.reporting._package_versions", return_value={"torch": "test"}
        ), patch("benchmarks.reporting.subprocess.run", side_effect=run):
            environment = collect_environment(Path.cwd(), "cpu")

        self.assertEqual(environment["commit"], revision)
        self.assertTrue(environment["tracked_dirty"])
        self.assertIn(["git", "status", "--porcelain", "--untracked-files=no"], commands)
        self.assertEqual(environment["nvidia_smi"]["output"], "first line\nsecond line")
        self.assertEqual(environment["nvcc"]["output"], "first line\nsecond line")
        json.dumps(environment, allow_nan=False)

    def test_gpu_metadata_uses_the_selected_device(self):
        fake_torch = self.fake_torch()
        fake_torch.device = lambda _: SimpleNamespace(type="cuda", index=2)
        properties = SimpleNamespace(
            name="selected GPU",
            uuid="abaa5fe0-76a5-6def-ee23-70d8993c1c66",
            major=8,
            minor=9,
            total_memory=24 * 1024**3,
            multi_processor_count=128,
        )
        requested_indices = []

        def gpu_properties(index):
            requested_indices.append(index)
            return properties

        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=gpu_properties,
        )
        fake_cv2 = SimpleNamespace(__version__="test", getNumThreads=lambda: 1)
        with patch.dict(sys.modules, {"torch": fake_torch, "cv2": fake_cv2}), patch(
            "benchmarks.reporting._package_versions", return_value={"torch": "test"}
        ), patch("benchmarks.reporting.subprocess.run", side_effect=FileNotFoundError):
            environment = collect_environment(Path.cwd(), "cuda:2")

        self.assertEqual(requested_indices, [2])
        self.assertEqual(
            environment["gpu"],
            {
                "available": True,
                "index": 2,
                "name": "selected GPU",
                "uuid": "abaa5fe0-76a5-6def-ee23-70d8993c1c66",
                "compute_capability": [8, 9],
                "total_memory_bytes": 24 * 1024**3,
                "multiprocessor_count": 128,
            },
        )
        json.dumps(environment, allow_nan=False)

    def test_gpu_uuid_is_none_when_properties_do_not_expose_it(self):
        fake_torch = self.fake_torch()
        fake_torch.device = lambda _: SimpleNamespace(type="cuda", index=0)
        properties = SimpleNamespace(
            name="GPU without UUID",
            major=8,
            minor=9,
            total_memory=24 * 1024**3,
            multi_processor_count=128,
        )
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _: properties,
        )
        fake_cv2 = SimpleNamespace(__version__="test", getNumThreads=lambda: 1)
        with patch.dict(sys.modules, {"torch": fake_torch, "cv2": fake_cv2}), patch(
            "benchmarks.reporting._package_versions", return_value={"torch": "test"}
        ), patch("benchmarks.reporting.subprocess.run", side_effect=FileNotFoundError), patch(
            "benchmarks.reporting.platform.node", return_value="benchmark-host"
        ):
            environment = collect_environment(Path.cwd(), "cuda:0")

        self.assertTrue(environment["gpu"]["available"])
        self.assertIsNone(environment["gpu"]["uuid"])
        self.assertEqual(environment["hostname"], "benchmark-host")
        json.dumps(environment, allow_nan=False)


# Support unittest discovery and direct execution without a GPU.
if __name__ == "__main__":
    unittest.main()
