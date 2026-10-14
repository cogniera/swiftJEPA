#Author : Paarth Sharma
#File Name : test_plot_results.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : CPU checks for measured latency plots and malformed result files.

# Verify actual figures and PNG output using disposable files inside the writable workspace.
import csv
from pathlib import Path
import tempfile
import unittest

from benchmarks.plot_results import STAGES, plot_run

import matplotlib.pyplot as plt
import numpy as np


# Deliberately separate component medians from the measured end-to-end median.
TEST_DIR = Path(__file__).resolve().parent
SAMPLES = [[6.0, 1.0, 1.0, 10.0], [1.0, 6.0, 1.0, 12.0], [1.0, 1.0, 6.0, 20.0]]


# Keep sample fixtures small while preserving the benchmark's real CSV and summary schemas.
class PlotResultsTests(unittest.TestCase):
    @staticmethod
    def write_csv(path, rows, fields=("iteration", *STAGES)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    @staticmethod
    def summary(passed=True):
        return {
            "config": {"model": "vitl", "frames": 16, "batch": 1, "dtype": "fp16", "compile": True},
            "stages": {"end_to_end_ms": {"count": 3, "median_ms": 12.0, "p95_ms": 19.2}},
            "validity": {"passed": passed, "reasons": [] if passed else ["requires at least 200 measured iterations"]},
        }

    def test_figure_uses_raw_e2e_statistics_and_independent_rows(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
            csv_path = Path(temporary) / "timings.csv"
            output_path = Path(temporary) / "figures" / "latency.png"
            self.write_csv(csv_path, [[index, *row] for index, row in enumerate(SAMPLES)])
            figure = plot_run(self.summary(), csv_path, output_path)

            self.assertTrue(output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertFalse(plt.fignum_exists(figure.number))
            image = figure.axes[0].images[0]
            np.testing.assert_allclose(image.get_array(), np.asarray(SAMPLES).T)
            labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
            self.assertEqual(labels[-1], "End to end (primary)")
            markers = figure.axes[1].lines
            np.testing.assert_allclose(markers[0].get_xdata(), [12.0, 12.0])
            np.testing.assert_allclose(markers[1].get_xdata(), [19.2, 19.2])
            legend = [label.get_text() for label in figure.axes[1].get_legend().get_texts()]
            self.assertEqual(legend, ["Median 12.000 ms", "p95 19.200 ms"])
            self.assertIn("256 px", figure.get_suptitle())
            self.assertIn("max-autotune | timing valid", figure.get_suptitle())
            self.assertIn("raw samples", " ".join(text.get_text() for text in figure.texts))

    # A failed timing gate must remain visible even when the input CSV is structurally valid.
    def test_rejected_run_is_labeled_and_eager_mode_is_reported(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
            csv_path = Path(temporary) / "timings.csv"
            self.write_csv(csv_path, [[index, *row] for index, row in enumerate(SAMPLES)])
            summary = self.summary(passed=False)
            summary["config"]["compile"] = False
            figure = plot_run(summary, csv_path, Path(temporary) / "latency.png")
            self.assertIn("eager | timing invalid", figure.get_suptitle())
            self.assertIn("requires at least 200", " ".join(text.get_text() for text in figure.texts))
            self.assertFalse(plt.fignum_exists(figure.number))

    # Reject unusable timing values before output creation or figure allocation.
    def test_nonfinite_nonpositive_and_missing_values_are_rejected(self):
        for value in (0, -1, "nan", "inf", "-inf", "invalid", ""):
            with self.subTest(value=value), tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
                csv_path = Path(temporary) / "timings.csv"
                self.write_csv(csv_path, [[0, 1, 1, 1, value]])
                output_path = Path(temporary) / "latency.png"
                before = plt.get_fignums()
                with self.assertRaises(ValueError):
                    plot_run({}, csv_path, output_path)
                self.assertEqual(plt.get_fignums(), before)
                self.assertFalse(output_path.exists())

    def test_empty_missing_stage_and_unordered_iteration_files_are_rejected(self):
        cases = (
            ([], ("iteration", *STAGES)),
            ([[0, 1, 1, 1]], ("iteration", *STAGES[:-1])),
            ([[0, 1, 1, 1, 2], [0, 1, 1, 1, 2]], ("iteration", *STAGES)),
            ([[-1, 1, 1, 1, 2]], ("iteration", *STAGES)),
        )
        for rows, fields in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
                csv_path = Path(temporary) / "timings.csv"
                self.write_csv(csv_path, rows, fields)
                with self.assertRaises(ValueError):
                    plot_run({}, csv_path, Path(temporary) / "latency.png")

    # A mismatched summary must never label a chart with another run's reported statistics.
    def test_stale_summary_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as temporary:
            csv_path = Path(temporary) / "timings.csv"
            self.write_csv(csv_path, [[index, *row] for index, row in enumerate(SAMPLES)])
            summary = self.summary()
            summary["stages"]["end_to_end_ms"]["median_ms"] = 3.0
            with self.assertRaisesRegex(ValueError, "do not match"):
                plot_run(summary, csv_path, Path(temporary) / "latency.png")


# Support direct execution and discovery from the repository root.
if __name__ == "__main__":
    unittest.main()
