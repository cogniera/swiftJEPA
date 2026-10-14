#Author : Paarth Sharma
#File Name : test_accuracy.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : CPU tests for per clip quality gates and deterministic test videos.

import json
import unittest

import numpy as np
import torch

from benchmarks.accuracy import compare_embeddings, iter_test_clips, validate_encoder, validate_pipelines


# Test metric failures independently of pretrained weights or CUDA availability.
class EmbeddingMetricTests(unittest.TestCase):
    def test_minimum_cosine_rejects_one_bad_batch_member(self):
        reference = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        candidate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = compare_embeddings(reference, candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["per_sample_cosine"], [1.0, 0.0])
        self.assertEqual(result["minimum_cosine"], 0.0)
        self.assertEqual(result["failure"], "cosine_below_threshold")

    def test_scale_error_is_reported_separately_from_cosine(self):
        reference = torch.tensor([[[3.0, 4.0]]])
        result = compare_embeddings(reference, reference * 2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["minimum_cosine"], 1.0)
        self.assertEqual(result["relative_l2"], 1.0)
        self.assertEqual(result["maximum_absolute_error"], 4.0)

    def test_nonfinite_values_fail_without_nonfinite_json(self):
        for invalid_value in (float("nan"), float("inf"), -float("inf")):
            for invalid_reference in (False, True):
                with self.subTest(value=invalid_value, invalid_reference=invalid_reference):
                    good = torch.tensor([[1.0, 2.0]])
                    bad = torch.tensor([[1.0, invalid_value]])
                    reference, candidate = (bad, good) if invalid_reference else (good, bad)
                    result = compare_embeddings(reference, candidate)
                    self.assertFalse(result["passed"])
                    self.assertEqual(result["failure"], "nonfinite_embeddings")
                    json.dumps(result, allow_nan=False)

    def test_shape_and_zero_norm_fail_explicitly(self):
        reference = torch.tensor([[1.0, 2.0]])
        mismatched = compare_embeddings(reference, reference.flatten())
        zero_norm = compare_embeddings(reference, torch.zeros_like(reference))
        self.assertEqual(mismatched["failure"], "shape_mismatch")
        self.assertEqual(zero_norm["failure"], "zero_norm_embeddings")


# Verify clip generation and sequential validation using tiny CPU encoder callables.
class EncoderValidationTests(unittest.TestCase):
    def test_noise_is_the_exact_timed_benchmark_input(self):
        from benchmarks.benchmark_vjepa2 import make_clip

        _, noise = next(iter_test_clips(frames=3, batch=2, crop_size=8, seed=17))
        np.testing.assert_array_equal(noise, make_clip(2, 3, 8, seed=17))

    @staticmethod
    def preprocess(clip):
        return torch.from_numpy(clip).permute(0, 4, 1, 2, 3).float().contiguous() / 255

    @staticmethod
    def encoder(value):
        channel_mean = value.mean(dim=(2, 3, 4))
        return torch.cat((channel_mean, torch.ones(value.shape[0], 1)), dim=1)

    def test_clip_suite_is_deterministic_and_has_temporal_variation(self):
        first = dict(iter_test_clips(frames=3, batch=2, crop_size=8, seed=17))
        second = dict(iter_test_clips(frames=3, batch=2, crop_size=8, seed=17))
        self.assertEqual(len(first), 7)
        for name, clip in first.items():
            with self.subTest(name=name):
                self.assertEqual(clip.shape, (2, 3, 8, 8, 3))
                self.assertIsInstance(clip, np.ndarray)
                self.assertEqual(clip.dtype, np.uint8)
                self.assertTrue(clip.flags.c_contiguous)
                self.assertTrue(np.array_equal(clip, second[name]))
                if name.startswith("moving_"):
                    self.assertFalse(np.array_equal(clip[:, 0], clip[:, 1]))
        self.assertTrue(np.array_equal(first["black"], np.zeros_like(first["black"])))
        self.assertTrue(np.array_equal(first["white"], np.full_like(first["white"], 255)))

    def test_one_bad_named_clip_fails_entire_suite(self):
        def candidate(value):
            output = self.encoder(value)
            if bool((value == 0).all()):
                output = -output
            return output

        result = validate_encoder(self.encoder, candidate, self.preprocess, "cpu", torch.float32,
                                  frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        self.assertEqual(result["minimum_cosine"], -1.0)
        self.assertEqual([row["name"] for row in result["clips"] if not row["passed"]], ["black"])
        json.dumps(result, allow_nan=False)

    def test_reference_snapshot_survives_shared_output_buffer(self):
        buffer = torch.empty(1, 4)

        def reference(value):
            buffer.copy_(self.encoder(value))
            return buffer

        def candidate(value):
            buffer.copy_(-self.encoder(value))
            return buffer

        result = validate_encoder(reference, candidate, self.preprocess, "cpu", torch.float32,
                                  frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        for row in result["clips"]:
            self.assertAlmostEqual(row["minimum_cosine"], -1.0)

    def test_repeated_input_failure_is_detected(self):
        call_count = 0

        def candidate(value):
            nonlocal call_count
            call_count += 1
            output = self.encoder(value)
            return -output if call_count % 2 == 0 else output

        result = validate_encoder(self.encoder, candidate, self.preprocess, "cpu", torch.float32,
                                  frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        self.assertTrue(all(row["passed"] for row in result["clips"]))
        self.assertTrue(all(not row["repeat_input"]["passed"] for row in result["clips"]))

    def test_stale_output_after_input_changes_is_detected(self):
        cached_output = None

        def candidate(value):
            nonlocal cached_output
            if cached_output is None:
                cached_output = self.encoder(value)
            return cached_output

        result = validate_encoder(self.encoder, candidate, self.preprocess, "cpu", torch.float32,
                                  frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        self.assertTrue(result["clips"][0]["passed"])
        self.assertFalse(result["clips"][1]["passed"])

    def test_matching_encoder_passes_every_clip(self):
        result = validate_encoder(self.encoder, self.encoder, self.preprocess, "cpu", torch.float32,
                                  frames=2, crop_size=8)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["clips"]), 7)
        self.assertAlmostEqual(result["minimum_cosine"], 1.0)


# Exercise separate raw preprocessing paths and native precision without CUDA.
class PipelineValidationTests(unittest.TestCase):
    @staticmethod
    def reference_pipeline(clip):
        return EncoderValidationTests.encoder(EncoderValidationTests.preprocess(clip))

    def test_native_pipeline_receives_raw_clips_without_ambient_autocast(self):
        observed_autocast = []

        def reference(clip):
            observed_autocast.append(torch.is_autocast_enabled("cpu"))
            return self.reference_pipeline(clip)

        def candidate(clip):
            self.assertIsInstance(clip, np.ndarray)
            self.assertEqual(clip.dtype, np.uint8)
            self.assertEqual(clip.shape, (1, 2, 8, 8, 3))
            observed_autocast.append(torch.is_autocast_enabled("cpu"))
            return self.reference_pipeline(clip).to(torch.bfloat16)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = validate_pipelines(reference, candidate, "cpu", torch.bfloat16,
                                        frames=2, crop_size=8, seed=17)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["clips"]), 7)
        self.assertFalse(any(observed_autocast))
        self.assertEqual(len(observed_autocast), 21)
        self.assertEqual(result["candidate_dtype"], "bfloat16")
        self.assertNotIn("candidate_autocast_dtype", result)
        self.assertEqual(result["validation_scope"], "raw_uint8_to_embeddings")
        self.assertEqual(result["clip_seed"], 17)
        json.dumps(result, allow_nan=False)

    def test_preprocessing_mismatch_fails_complete_pipeline_gate(self):
        def candidate(clip):
            return self.reference_pipeline(255 - clip)

        result = validate_pipelines(self.reference_pipeline, candidate, "cpu", torch.float32,
                                    frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        black = next(row for row in result["clips"] if row["name"] == "black")
        self.assertAlmostEqual(black["minimum_cosine"], 0.5)

    def test_raw_pipeline_reference_snapshot_survives_shared_buffer(self):
        buffer = torch.empty(1, 4)

        def reference(clip):
            buffer.copy_(self.reference_pipeline(clip))
            return buffer

        def candidate(clip):
            buffer.copy_(-self.reference_pipeline(clip))
            return buffer

        result = validate_pipelines(reference, candidate, "cpu", torch.float32,
                                    frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        for row in result["clips"]:
            self.assertAlmostEqual(row["minimum_cosine"], -1.0)

    def test_raw_pipeline_repeated_input_corruption_is_detected(self):
        call_count = 0

        def candidate(clip):
            nonlocal call_count
            call_count += 1
            output = self.reference_pipeline(clip)
            return -output if call_count % 2 == 0 else output

        result = validate_pipelines(self.reference_pipeline, candidate, "cpu", torch.float32,
                                    frames=2, crop_size=8)
        self.assertFalse(result["passed"])
        self.assertTrue(all(row["passed"] for row in result["clips"]))
        self.assertTrue(all(not row["repeat_input"]["passed"] for row in result["clips"]))


# Run with unittest discovery so importing the module does not execute tests.
if __name__ == "__main__":
    unittest.main()
