#Author : Paarth Sharma
#File Name : test_pipeline.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Tests for fidelity to the official raw video input pipeline

"""Run with python -m unittest discover -s benchmarks/tests -p test_pipeline.py."""

# Use structured frames to expose spatial, temporal, and channel ordering errors.
import unittest

import numpy as np
import torch

from benchmarks.pipeline import make_preprocess


def structured_clip(batch, frames, height, width):
    time, y, x = np.indices((frames, height, width), dtype=np.int32)
    channels = (
        (3 * x + y + 11 * time) % 256,
        (2 * y + 7 * time) % 256,
        ((x // 3 + y // 5 + time) % 2) * 255,
    )
    clip = np.stack(channels, axis=-1).astype(np.uint8)
    return np.stack([np.roll(clip, 5 * item, axis=2) for item in range(batch)])


# Compare the batch adapter against upstream single-view transforms at target size.
class OfficialPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocess = staticmethod(make_preprocess(256))
        from evals.video_classification_frozen.utils import make_transforms

        cls.official = make_transforms(
            training=False, num_views_per_clip=1, crop_size=256
        )

    def test_structured_clips_match_upstream(self):
        for height, width in ((256, 256), (180, 300), (301, 171)):
            with self.subTest(height=height, width=width):
                clip = structured_clip(2, 3, height, width)
                unchanged = clip.copy()
                expected = torch.stack([self.official(item)[0] for item in clip])
                actual = self.preprocess(clip)

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                np.testing.assert_array_equal(clip, unchanged)
                self.assertEqual(actual.shape, (2, 3, 3, 256, 256))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")
                self.assertTrue(actual.is_contiguous())

    def test_target_workload_keeps_all_frames(self):
        clip = structured_clip(1, 16, 256, 256)
        actual = self.preprocess(clip)
        expected = self.official(clip[0])[0].unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (1, 3, 16, 256, 256))

    def test_resize_and_center_crop_change_square_input(self):
        clip = structured_clip(1, 2, 256, 256)
        normalized_only = torch.from_numpy(clip).float().div(255).permute(0, 4, 1, 2, 3)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        normalized_only = (normalized_only - mean) / std
        self.assertFalse(torch.allclose(self.preprocess(clip), normalized_only))

    def test_noncontiguous_input_matches_upstream(self):
        clip = structured_clip(1, 2, 40, 70)[:, :, :, ::-1, :]
        self.assertFalse(clip.flags.c_contiguous)
        expected = self.official(clip[0])[0].unsqueeze(0)
        torch.testing.assert_close(self.preprocess(clip), expected, rtol=0, atol=0)

    # Constant colors provide an independent check of channel normalization values.
    def test_constant_colors_use_expected_normalization(self):
        clip = np.empty((1, 2, 32, 48, 3), dtype=np.uint8)
        clip[...] = (0, 127, 255)
        actual = make_preprocess(32)(clip)
        values = torch.tensor((0, 127, 255)).float().div(255)
        expected = (values - torch.tensor((0.485, 0.456, 0.406))) / torch.tensor(
            (0.229, 0.224, 0.225)
        )
        expected = expected.view(1, 3, 1, 1, 1).expand_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_custom_crop_size_matches_upstream(self):
        from evals.video_classification_frozen.utils import make_transforms

        clip = structured_clip(1, 3, 31, 53)
        transform = make_transforms(training=False, num_views_per_clip=1, crop_size=32)
        actual = make_preprocess(32)(clip)
        expected = transform(clip[0])[0].unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (1, 3, 3, 32, 32))

    # Reject malformed clips before invoking the upstream transform.
    def test_rejects_invalid_clips(self):
        invalid = (
            ([], TypeError),
            (torch.zeros(1, 2, 32, 32, 3, dtype=torch.uint8), TypeError),
            (np.zeros((1, 2, 32, 32, 3), dtype=np.float32), TypeError),
            (np.zeros((2, 32, 32, 3), dtype=np.uint8), ValueError),
            (np.zeros((1, 2, 32, 32, 1), dtype=np.uint8), ValueError),
        )
        for clip, exception in invalid:
            with self.subTest(type=type(clip), shape=getattr(clip, "shape", None)):
                with self.assertRaises(exception):
                    self.preprocess(clip)
        for axis in range(4):
            shape = [1, 2, 32, 32, 3]
            shape[axis] = 0
            with self.subTest(empty_axis=axis):
                with self.assertRaises(ValueError):
                    self.preprocess(np.empty(shape, dtype=np.uint8))

    def test_rejects_invalid_crop_size(self):
        for size in (True, 32.0, "32"):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    make_preprocess(size)
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    make_preprocess(size)


# Keep the test file runnable through the standard unittest entry point.
if __name__ == "__main__":
    unittest.main()
