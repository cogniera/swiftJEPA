#Author : Paarth Sharma
#File Name : plot_results.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Static latency heatmaps and distributions from measured iteration samples.

# Use a noninteractive backend for disposable benchmark machines and CPU verification.
import csv
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Keep independently measured intervals in a fixed order, with the primary metric last.
STAGES = ("preprocess_cpu_ms", "h2d_copy_ms", "encoder_cuda_ms", "end_to_end_ms")
STAGE_LABELS = ("CPU preprocessing", "Host to device", "Encoder CUDA", "End to end (primary)")


# Reject incomplete, unordered, or unusable samples before drawing a figure.
def _read_samples(csv_path):
    iterations, samples = [], []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"iteration", *STAGES}.issubset(reader.fieldnames):
            raise ValueError("CSV requires iteration and all four latency stage columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                iteration = int(row["iteration"])
                values = [float(row[stage]) for stage in STAGES]
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV row {row_number} contains invalid timing values") from error
            if iteration < 0 or (iterations and iteration <= iterations[-1]):
                raise ValueError("CSV iteration indices must be nonnegative and strictly increasing")
            if None in row or not np.isfinite(values).all() or any(value <= 0 for value in values):
                raise ValueError(f"CSV row {row_number} requires finite positive latency values")
            iterations.append(iteration)
            samples.append(values)
    if not samples:
        raise ValueError("CSV must contain at least one measured iteration")
    return iterations, np.asarray(samples, dtype=np.float64).T


# Derive E2E statistics from the raw samples and reject a stale summary when provided.
def _end_to_end_statistics(summary, samples):
    median = float(np.median(samples))
    p95 = float(np.percentile(samples, 95))
    stored = summary.get("stages", {}).get("end_to_end_ms")
    if stored is not None:
        try:
            consistent = (
                stored["count"] == len(samples)
                and np.isclose(float(stored["median_ms"]), median, rtol=1e-6, atol=1e-9)
                and np.isclose(float(stored["p95_ms"]), p95, rtol=1e-6, atol=1e-9)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Summary contains invalid end-to-end statistics") from error
        if not consistent:
            raise ValueError("Summary end-to-end statistics do not match the CSV samples")
    return median, p95


# Save a compact PNG and return its closed Figure for inspection of labels and plotted values.
def plot_run(summary, csv_path, output_path):
    iterations, samples = _read_samples(csv_path)
    end_to_end = samples[-1]
    median, p95 = _end_to_end_statistics(summary, end_to_end)
    config = summary.get("config", {})
    model = config.get("model", "encoder")
    crop_size = config.get("crop_size", 384 if model.endswith("-384") else 256)
    mode = "max-autotune" if config.get("compile", False) else "eager"
    validity = summary.get("validity", {})
    passed = validity.get("passed")
    status = "valid" if passed is True else "invalid" if passed is False else "unverified"
    title = (
        f"{model} | {config.get('frames', '?')} frames | {crop_size} px | "
        f"batch {config.get('batch', '?')} | {config.get('dtype', '?')}\n"
        f"{mode} | timing {status} | {len(iterations)} measured iterations"
    )

    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, (heatmap, distribution) = plt.subplots(
            1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.55, 1]}
        )
        try:
            figure.subplots_adjust(left=0.18, right=0.98, top=0.75, bottom=0.22, wspace=0.48)
            figure.suptitle(title, fontsize=11, y=0.96)

            # A shared absolute scale preserves the difference between components and the primary metric.
            image = heatmap.imshow(samples, aspect="auto", interpolation="nearest", cmap="cividis")
            heatmap.set_yticks(range(len(STAGES)), labels=STAGE_LABELS)
            heatmap.get_yticklabels()[-1].set_fontweight("bold")
            ticks = np.unique(np.linspace(0, len(iterations) - 1, min(5, len(iterations)), dtype=int))
            heatmap.set_xticks(ticks, labels=[iterations[index] for index in ticks])
            heatmap.set_xlabel("Measured iteration")
            heatmap.set_title("Independent measured intervals", fontsize=10)
            colorbar = figure.colorbar(image, ax=heatmap, fraction=0.05, pad=0.035)
            colorbar.set_label("Latency (ms)")

            # Compute percentile markers directly from E2E samples, never from component medians.
            bins = max(1, min(30, int(np.sqrt(len(end_to_end)))))
            distribution.hist(end_to_end, bins=bins, color="#477b91", edgecolor="white")
            distribution.axvline(median, color="#ad4e21", linestyle="--", label=f"Median {median:.3f} ms")
            distribution.axvline(p95, color="#67432d", linestyle=":", label=f"p95 {p95:.3f} ms")
            distribution.set_title("End-to-end distribution", fontsize=10)
            distribution.set_xlabel("End-to-end latency (ms)")
            distribution.set_ylabel("Iterations")
            distribution.legend(
                loc="best", fontsize=8, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95
            )
            distribution.grid(axis="y", alpha=0.2)

            figure.text(
                0.18, 0.08,
                "Rows are independent measurements; end-to-end statistics come from raw samples.",
                fontsize=8,
            )
            if passed is False and validity.get("reasons"):
                reason = textwrap.fill("Rejected: " + "; ".join(validity["reasons"]), width=115)
                figure.text(0.18, 0.025, reason, fontsize=7, color="#8e3427", va="bottom")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=160, format="png")
        finally:
            plt.close(figure)
    return figure
