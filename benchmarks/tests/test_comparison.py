#Author : Paarth Sharma
#File Name : test_comparison.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Synthetic schema tests for provenance quality and lossless latency gates.

from copy import deepcopy
import json
import unittest

from benchmarks.comparison import compare_lossless_runs


# These invented fixture values test verdict boundaries and are not benchmark results.
def make_summary(backend, median_ms, dtype="fp16"):
    accuracy = {
        "passed": True, "reference_dtype": "float32", "minimum_cosine": 0.999,
        "candidate_autocast_dtype": "float16" if dtype == "fp16" else "bfloat16",
        "frames": 16, "batch": 1, "crop_size": 256, "clip_seed": 0,
        "clip_layout": "BTHWC uint8", "clip_suite_version": 1,
        "clips": [{"name": name, "passed": True, "minimum_cosine": 0.999,
                   "repeat_input": {"passed": True, "minimum_cosine": 0.999}}
                  for name in ("noise", "black", "white", "mid_gray", "moving_gradient",
                               "moving_checkerboard", "moving_color_bars")],
    }
    return {
        "config": {"backend": backend, "model": "vitl", "frames": 16, "batch": 1,
                   "crop_size": 256, "dtype": dtype, "compile": backend == "official",
                   "pretrained": True, "seed": 0, "cpu_threads": 4, "opencv_threads": 1,
                   "session_id": "synthetic-test-session"},
        "compilation": {"enabled": backend == "official", "mode": "max-autotune"},
        "environment": {"commit": "a" * 40, "tracked_dirty": False,
                        "gpu": {"name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-test"},
                        "packages": {"torch": "test-torch"}, "torch_cuda_version": "test-cuda"},
        "metric": "raw pageable uint8 BTHWC to device embeddings using synchronized CUDA events",
        "preprocess": "official single view resize center crop tensor conversion normalization and pinning",
        "input": {"layout": "BTHWC", "dtype": "uint8", "memory": "pageable CPU",
                  "shape": [1, 16, 256, 256, 3], "crop_size": 256,
                  "short_side_resize": 292, "sha256": "0" * 64},
        "stages": {"end_to_end_ms": {"median_ms": median_ms, "mean_ms": median_ms,
                                      "p95_ms": median_ms, "count": 200}},
        "validity": {"passed": True, "reasons": [], "p95_mean_ratio": 1.0},
        "accuracy": accuracy,
    }


# Test independent rejection paths so fast medians cannot override missing evidence.
class LosslessComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = make_summary("official", 18.0)
        self.engine = make_summary("swiftjepa", 12.0)

    def verdict(self):
        return compare_lossless_runs(self.baseline, self.engine)

    def test_exact_thresholds_pass_for_both_lossless_dtypes(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                result = compare_lossless_runs(self.baseline, make_summary("swiftjepa", 12.0, dtype))
                self.assertTrue(result["passed"])
                self.assertEqual(result["speedup"], 1.5)
                self.assertEqual(result["measurements"]["latency_reduction_ms"], 6.0)
                self.assertEqual(result["reasons"], [])
                json.dumps(result, allow_nan=False)

    def test_native_precision_metadata_replaces_autocast_metadata(self):
        for dtype, native_dtype in (("fp16", "float16"), ("bf16", "bfloat16")):
            engine = make_summary("swiftjepa", 12.0, dtype)
            del engine["accuracy"]["candidate_autocast_dtype"]
            engine["accuracy"]["candidate_dtype"] = native_dtype
            self.assertTrue(compare_lossless_runs(self.baseline, engine)["passed"])
        self.engine["accuracy"]["candidate_dtype"] = "bfloat16"
        self.assertFalse(self.verdict()["passed"])

    def test_actual_quantization_formats_are_not_lossless_results(self):
        for disabled in (None, "none"):
            self.engine["config"]["quant"] = disabled
            self.assertTrue(self.verdict()["passed"])
        for side in ("baseline", "engine"):
            for quant in ("fp8", "int8", "int4"):
                with self.subTest(side=side, quant=quant):
                    baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                    summary = baseline if side == "baseline" else engine
                    summary["config"]["quant"] = quant
                    self.assertFalse(compare_lossless_runs(baseline, engine)["passed"])

    def test_each_performance_target_is_independent(self):
        self.baseline["stages"]["end_to_end_ms"]["median_ms"] = 17.999
        result = self.verdict()
        self.assertFalse(result["passed"])
        self.assertTrue(result["comparable"])
        self.assertEqual(result["target_reasons"], ["engine speedup must be at least 1.5"])
        self.baseline = make_summary("official", 19.0)
        self.engine = make_summary("swiftjepa", 12.001)
        self.assertEqual(self.verdict()["target_reasons"], ["engine end to end median must be at most 12 ms"])

    def test_baseline_accuracy_failure_or_skip_does_not_block_comparator(self):
        self.baseline["accuracy"]["passed"] = False
        self.baseline["accuracy"]["minimum_cosine"] = 0.98
        self.baseline["accuracy"]["clips"][0]["minimum_cosine"] = 0.98
        self.assertTrue(self.verdict()["passed"])
        self.baseline["accuracy"] = {"passed": None, "reason": "not validated"}
        self.assertTrue(self.verdict()["passed"])

    def test_skipped_baseline_accuracy_still_requires_the_complete_engine_suite(self):
        self.baseline["accuracy"] = {"passed": None, "reason": "not validated"}
        self.engine["accuracy"]["clips"] = self.engine["accuracy"]["clips"][:-1]
        result = self.verdict()
        self.assertFalse(result["passed"])
        self.assertFalse(result["accuracy_passed"])
        self.assertTrue(any("moving_color_bars" in reason for reason in result["accuracy_reasons"]))
        self.engine["accuracy"]["clips"] = [{"name": "arbitrary", "passed": True,
                                              "minimum_cosine": 1.0,
                                              "repeat_input": {"passed": True, "minimum_cosine": 1.0}}]
        self.assertFalse(self.verdict()["passed"])

    def test_legacy_baseline_backend_requires_compiled_mode(self):
        del self.baseline["config"]["backend"]
        self.assertTrue(self.verdict()["passed"])
        self.baseline["compilation"]["mode"] = "default"
        self.assertFalse(self.verdict()["passed"])

    def test_provenance_mismatches_retain_unusable_measured_difference(self):
        mutations = (
            ("session", lambda summary: summary["config"].update(session_id="another-session")),
            ("GPU", lambda summary: summary["environment"]["gpu"].update(uuid="GPU-other")),
            ("input seed", lambda summary: summary["config"].update(seed=1)),
            ("CPU threads", lambda summary: summary["config"].update(cpu_threads=8)),
            ("OpenCV threads", lambda summary: summary["config"].update(opencv_threads=2)),
            ("torch", lambda summary: summary["environment"]["packages"].update(torch="different")),
            ("CUDA", lambda summary: summary["environment"].update(torch_cuda_version="different")),
            ("raw semantics", lambda summary: summary.update(metric="encoder only CUDA events")),
            ("raw clip bytes", lambda summary: summary["input"].update(sha256="1" * 64)),
            ("clip suite", lambda summary: summary["accuracy"]["clips"][1].update(name="white")),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                engine = deepcopy(self.engine)
                mutate(engine)
                result = compare_lossless_runs(self.baseline, engine)
                self.assertFalse(result["passed"])
                self.assertFalse(result["comparable"])
                self.assertEqual(result["speedup"], 1.5)
                self.assertTrue(result["incompatibility_reasons"])

    def test_target_workload_and_backend_mismatches_fail(self):
        mutations = (
            lambda summary: summary["config"].update(model="vith"),
            lambda summary: summary["config"].update(frames=32),
            lambda summary: summary["config"].update(batch=2),
            lambda summary: summary["config"].update(crop_size=384),
            lambda summary: summary["config"].update(pretrained=False),
            lambda summary: summary["config"].update(dtype="int8"),
            lambda summary: summary["config"].update(backend="official"),
            lambda summary: summary["environment"]["gpu"].update(name="NVIDIA RTX 3090"),
        )
        for mutate in mutations:
            engine = deepcopy(self.engine)
            mutate(engine)
            self.assertFalse(compare_lossless_runs(self.baseline, engine)["passed"])

    def test_missing_provenance_is_not_a_match(self):
        for section, key in (("config", "session_id"), ("config", "seed"),
                             ("environment", "torch_cuda_version")):
            with self.subTest(section=section, key=key):
                baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                del baseline[section][key]
                del engine[section][key]
                self.assertFalse(compare_lossless_runs(baseline, engine)["passed"])
        del self.baseline["environment"]["gpu"]["uuid"]
        del self.engine["environment"]["gpu"]["uuid"]
        self.assertFalse(self.verdict()["passed"])

    def test_missing_or_malformed_raw_clip_hash_cannot_pass(self):
        for digest in (None, "", "same_unknown_clip", "z" * 64):
            with self.subTest(digest=digest):
                baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                baseline["input"]["sha256"] = engine["input"]["sha256"] = digest
                self.assertFalse(compare_lossless_runs(baseline, engine)["passed"])
        del self.baseline["input"]
        del self.engine["input"]
        self.assertFalse(self.verdict()["passed"])

    def test_matching_incomplete_or_wrong_input_descriptors_do_not_qualify(self):
        for descriptor in ({"sha256": "0" * 64},
                           {**self.engine["input"], "shape": [1, 16, 384, 384, 3]},
                           {**self.engine["input"], "memory": "CUDA device"},
                           {**self.engine["input"], "short_side_resize": 256}):
            with self.subTest(descriptor=descriptor):
                baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                baseline["input"] = deepcopy(descriptor)
                engine["input"] = deepcopy(descriptor)
                result = compare_lossless_runs(baseline, engine)
                self.assertFalse(result["passed"])
                self.assertFalse(result["comparable"])

    def test_one_bad_clip_or_repeat_fails_despite_aggregate_pass(self):
        for repeated in (False, True):
            engine = deepcopy(self.engine)
            metrics = engine["accuracy"]["clips"][1]
            if repeated:
                metrics = metrics["repeat_input"]
            metrics["minimum_cosine"] = 0.998999
            result = compare_lossless_runs(self.baseline, engine)
            self.assertFalse(result["passed"])
            self.assertFalse(result["accuracy_passed"])
            self.assertTrue(result["performance_targets_met"])

    def test_nonfinite_quality_and_wrong_reference_fail(self):
        self.engine["accuracy"]["clips"][0]["minimum_cosine"] = float("nan")
        self.engine["accuracy"]["reference_dtype"] = "float16"
        result = self.verdict()
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["accuracy_reasons"]), 2)
        json.dumps(result, allow_nan=False)

    def test_noisy_or_invalid_latency_is_not_comparable(self):
        for label in ("baseline", "engine"):
            with self.subTest(label=label):
                baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                summary = baseline if label == "baseline" else engine
                summary["validity"]["passed"] = False
                summary["validity"]["reasons"] = ["p95 to mean ratio exceeds 1.3"]
                summary["validity"]["p95_mean_ratio"] = 1.31
                result = compare_lossless_runs(baseline, engine)
                self.assertFalse(result["passed"])
                self.assertFalse(result["comparable"])
        self.engine["validity"]["p95_mean_ratio"] = 1.31
        self.assertFalse(self.verdict()["passed"])

    def test_missing_count_or_tail_evidence_cannot_be_replaced_by_passed_flag(self):
        for section, key in (("stage", "count"), ("stage", "mean_ms"),
                             ("stage", "p95_ms"), ("validity", "p95_mean_ratio")):
            with self.subTest(section=section, key=key):
                baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                target = engine["stages"]["end_to_end_ms"] if section == "stage" else engine["validity"]
                del target[key]
                result = compare_lossless_runs(baseline, engine)
                self.assertFalse(result["passed"])
                self.assertFalse(result["comparable"])
        del self.engine["stages"]["end_to_end_ms"]["count"]
        self.engine["validity"]["sample_count"] = 200
        self.assertTrue(self.verdict()["passed"])

    def test_invalid_or_contradictory_tail_ratio_fails(self):
        for ratio in (None, 0, float("nan"), float("inf"), 1.01):
            with self.subTest(ratio=ratio):
                engine = deepcopy(self.engine)
                engine["validity"]["p95_mean_ratio"] = ratio
                result = compare_lossless_runs(self.baseline, engine)
                self.assertFalse(result["passed"])
                json.dumps(result, allow_nan=False)

    def test_dirty_missing_or_mismatched_source_cannot_pass(self):
        for side in ("baseline", "engine"):
            for dirty in (True, None):
                with self.subTest(side=side, dirty=dirty):
                    baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                    summary = baseline if side == "baseline" else engine
                    summary["environment"]["tracked_dirty"] = dirty
                    self.assertFalse(compare_lossless_runs(baseline, engine)["passed"])
            for commit in (None, "", "a" * 7, "z" * 40):
                with self.subTest(side=side, commit=commit):
                    baseline, engine = deepcopy(self.baseline), deepcopy(self.engine)
                    summary = baseline if side == "baseline" else engine
                    summary["environment"]["commit"] = commit
                    self.assertFalse(compare_lossless_runs(baseline, engine)["passed"])
        self.engine["environment"]["commit"] = "b" * 40
        result = self.verdict()
        self.assertFalse(result["passed"])
        self.assertIn("runs must use the same source commit", result["incompatibility_reasons"])

    def test_invalid_medians_never_generate_nonfinite_json(self):
        for value in (0, -1, True, "12", float("nan"), float("inf"), None):
            with self.subTest(value=value):
                self.engine["stages"]["end_to_end_ms"]["median_ms"] = value
                result = self.verdict()
                self.assertFalse(result["passed"])
                self.assertIsNone(result["speedup"])
                json.dumps(result, allow_nan=False)
        self.baseline["stages"]["end_to_end_ms"]["median_ms"] = 1e308
        self.engine["stages"]["end_to_end_ms"]["median_ms"] = 1e-308
        result = self.verdict()
        self.assertFalse(result["passed"])
        json.dumps(result, allow_nan=False)

    def test_empty_summaries_and_input_objects_are_not_mutated(self):
        original_baseline, original_engine = deepcopy(self.baseline), deepcopy(self.engine)
        self.verdict()
        self.assertEqual(self.baseline, original_baseline)
        self.assertEqual(self.engine, original_engine)
        result = compare_lossless_runs({}, {})
        self.assertFalse(result["passed"])
        json.dumps(result, allow_nan=False)


# Run the comparison tests without importing any GPU or numerical dependencies.
if __name__ == "__main__":
    unittest.main()
