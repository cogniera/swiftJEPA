#Author : Paarth Sharma
#File Name : test_profiling.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : CPU checks for real operator traces and failed profiling pass cleanup.

# Exercise actual PyTorch profiling without weights or a CUDA dependency.
import json
from pathlib import Path
import tempfile
import unittest

import torch

from benchmarks.profiling import profile_callable


# Keep disposable exports inside the writable workspace on restricted Windows hosts.
TEST_DIR = Path(__file__).resolve().parent


# Validate exported operation names and shapes as well as the callable's numerical output.
class ProfilingTests(unittest.TestCase):
    def test_cpu_profile_exports_one_complete_pipeline_call(self):
        matrix = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 255
        expected = matrix @ matrix.T
        outputs = []

        def run_once():
            result = matrix @ matrix.T
            outputs.append(result)
            return result

        with tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
            out_dir = Path(temporary) / "diagnostics"
            metadata = profile_callable(run_once, out_dir, use_cuda=False)
            trace = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
            top_ops = (out_dir / "top_ops.txt").read_text(encoding="utf-8")

            self.assertEqual(len(outputs), 1)
            torch.testing.assert_close(outputs[0], expected)
            self.assertEqual(metadata["activities"], ["CPU"])
            self.assertEqual(metadata["sort_by"], "cpu_time_total")
            self.assertEqual(metadata["profiled_iterations"], 1)
            self.assertTrue(metadata["separate_from_latency"])
            self.assertTrue(metadata["record_shapes"])
            self.assertGreater(metadata["operator_count"], 0)
            self.assertEqual(Path(metadata["trace"]), out_dir / "trace.json")
            self.assertEqual(Path(metadata["top_ops"]), out_dir / "top_ops.txt")
            json.dumps(metadata, allow_nan=False)

            operation_names = {event["name"] for event in trace["traceEvents"]}
            self.assertIn("raw_clip_to_embeddings", operation_names)
            self.assertIn("aten::mm", operation_names)
            self.assertTrue(
                any(
                    event["name"] == "aten::mm"
                    and event.get("args", {}).get("Input Dims") == [[16, 16], [16, 16]]
                    for event in trace["traceEvents"]
                )
            )
            self.assertIn("aten::mm", top_ops)
            self.assertIn("Input Shapes", top_ops)

    # A failed callable must release the profiler so another independent pass can succeed.
    def test_failed_pipeline_propagates_and_profiler_can_be_reused(self):
        matrix = torch.eye(4)

        def fail():
            matrix @ matrix
            raise RuntimeError("pipeline failed")

        with tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
            out_dir = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
                profile_callable(fail, out_dir, use_cuda=False)
            self.assertFalse((out_dir / "trace.json").exists())
            self.assertFalse((out_dir / "top_ops.txt").exists())

            metadata = profile_callable(lambda: matrix @ matrix, out_dir, use_cuda=False)
            self.assertGreater(metadata["operator_count"], 0)
            self.assertIn(
                "aten::mm", (out_dir / "top_ops.txt").read_text(encoding="utf-8")
            )

    # Reject an unusable callable before creating directories or starting profiler state.
    def test_invalid_callable_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
            out_dir = Path(temporary) / "unused"
            with self.assertRaisesRegex(TypeError, "run_once must be callable"):
                profile_callable(None, out_dir, use_cuda=False)
            self.assertFalse(out_dir.exists())


# Support direct unittest execution as well as discovery from the repository root.
if __name__ == "__main__":
    unittest.main()
