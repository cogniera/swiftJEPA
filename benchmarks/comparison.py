#Author : Paarth Sharma
#File Name : comparison.py
#Project Name : swiftJEPA
#Creation Date : 14th October 2026
#Modification Date : 14th October 2026
#Description : Compare lossless engine latency against a matching compiled baseline.

from collections.abc import Mapping
import math


# Keep the required corpus explicit so skipped baseline validation cannot shrink it.
REQUIRED_CLIP_NAMES = frozenset({
    "noise", "black", "white", "mid_gray", "moving_gradient",
    "moving_checkerboard", "moving_color_bars",
})


# Treat incomplete saved summaries as failed evidence instead of raising key errors.
def _mapping(value):
    return value if isinstance(value, Mapping) else {}


# Reject missing metadata sentinels as well as blank strings.
def _known_text(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in {
        "unavailable", "unknown", "none", "null"
    }


# Require full recorded digests rather than short or unavailable source identifiers.
def _hex_digest(value, length):
    return isinstance(value, str) and len(value) == length and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


# Restrict reported numbers to finite JSON numbers and keep booleans out of metrics.
def _number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    if not math.isfinite(value) or (positive and value <= 0):
        return None
    return value


# Verify target workload metadata without accepting boolean values as integers.
def _target_config(config, accuracy, label, reasons):
    for key, expected in (("model", "vitl"), ("frames", 16), ("batch", 1), ("crop_size", 256)):
        value = config.get(key, accuracy.get(key) if key == "crop_size" else None)
        integer_expected = isinstance(expected, int)
        if value != expected or integer_expected and (isinstance(value, bool) or not isinstance(value, int)):
            reasons.append(f"{label} {key} must be {expected}")
    if config.get("pretrained") is not True:
        reasons.append(f"{label} must use pretrained weights")


# Retain observed medians even when another gate makes their comparison unusable.
def _measurements(baseline_stage, engine_stage, reasons):
    baseline = _number(baseline_stage.get("median_ms"), positive=True)
    engine = _number(engine_stage.get("median_ms"), positive=True)
    result = {"baseline_median_ms": baseline, "engine_median_ms": engine,
              "speedup": None, "latency_reduction_ms": None, "latency_reduction_percent": None}
    if baseline is None or engine is None:
        reasons.append("both end to end medians must be positive finite numbers")
        return result
    speedup = baseline / engine
    reduction = baseline - engine
    percent = reduction / baseline * 100
    result.update(speedup=speedup if math.isfinite(speedup) else None,
                  latency_reduction_ms=reduction,
                  latency_reduction_percent=percent if math.isfinite(percent) else None)
    if result["speedup"] is None:
        reasons.append("median latency ratio must be finite")
    return result


# Require every candidate clip and its repeated call to meet the fp32 quality gate.
def _accuracy_reasons(accuracy, config):
    reasons = []
    if accuracy.get("passed") is not True:
        reasons.append("engine accuracy validation must pass")
    if accuracy.get("reference_dtype") != "float32":
        reasons.append("engine accuracy reference must use float32")
    for key, expected in (("frames", 16), ("batch", 1), ("crop_size", 256)):
        if accuracy.get(key) != expected or isinstance(accuracy.get(key), bool):
            reasons.append(f"engine accuracy {key} must be {expected}")
    if accuracy.get("clip_seed") != config.get("seed") or isinstance(accuracy.get("clip_seed"), bool):
        reasons.append("engine accuracy clip seed must match its latency input seed")
    if accuracy.get("clip_layout") != "BTHWC uint8":
        reasons.append("engine accuracy clips must use raw BTHWC uint8 input")
    version = accuracy.get("clip_suite_version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        reasons.append("engine accuracy clip suite version is unavailable")
    measured_dtype = config.get("dtype")
    expected_dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(measured_dtype) if isinstance(measured_dtype, str) else None
    native_dtype = accuracy.get("candidate_dtype")
    recorded_dtype = native_dtype if native_dtype is not None else accuracy.get("candidate_autocast_dtype")
    if isinstance(recorded_dtype, str):
        recorded_dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(recorded_dtype, recorded_dtype)
    if expected_dtype is None or recorded_dtype != expected_dtype:
        reasons.append("engine accuracy dtype must match the measured lossless dtype")
    clips = accuracy.get("clips")
    if not isinstance(clips, list) or not clips:
        return [*reasons, "engine accuracy must include named test clips"]
    names = set()
    for index, raw_row in enumerate(clips):
        row = _mapping(raw_row)
        name = row.get("name")
        if not _known_text(name) or name in names:
            reasons.append(f"engine accuracy clip {index} has a missing or duplicate name")
        else:
            names.add(name)
        for description, metrics in (("first input", row), ("repeat input", _mapping(row.get("repeat_input")))):
            cosine = _number(metrics.get("minimum_cosine"))
            if metrics.get("passed") is not True or cosine is None or not 0.999 <= cosine <= 1:
                reasons.append(f"engine accuracy clip {index} {description} must have cosine at least 0.999")
    missing_names = REQUIRED_CLIP_NAMES - names
    if missing_names:
        reasons.append(f"engine accuracy is missing required clips {', '.join(sorted(missing_names))}")
    summary_cosine = _number(accuracy.get("minimum_cosine"))
    if summary_cosine is None or not 0.999 <= summary_cosine <= 1:
        reasons.append("engine minimum cosine must be at least 0.999")
    return reasons


# Check provenance before interpreting observed latency differences as target results.
def compare_lossless_runs(baseline_summary, engine_summary):
    """Return a JSON serializable verdict; baseline accuracy is not an acceptance gate."""
    baseline_summary, engine_summary = _mapping(baseline_summary), _mapping(engine_summary)
    baseline_config = _mapping(baseline_summary.get("config"))
    engine_config = _mapping(engine_summary.get("config"))
    baseline_accuracy = _mapping(baseline_summary.get("accuracy"))
    engine_accuracy = _mapping(engine_summary.get("accuracy"))
    baseline_environment = _mapping(baseline_summary.get("environment"))
    engine_environment = _mapping(engine_summary.get("environment"))
    reasons = []
    _target_config(baseline_config, baseline_accuracy, "baseline", reasons)
    _target_config(engine_config, engine_accuracy, "engine", reasons)
    if baseline_config.get("dtype") != "fp16":
        reasons.append("baseline dtype must be fp16")
    if engine_config.get("dtype") not in ("fp16", "bf16"):
        reasons.append("engine dtype must be fp16 or bf16")
    compilation = _mapping(baseline_summary.get("compilation"))
    if baseline_config.get("compile") is not True or compilation.get("enabled") is not True:
        reasons.append("baseline must compile the official encoder")
    if compilation.get("mode") != "max-autotune":
        reasons.append("baseline compilation mode must be max-autotune")
    if baseline_config.get("backend", "official") != "official":
        reasons.append("baseline backend must be official")
    if engine_config.get("backend") != "swiftjepa":
        reasons.append("engine backend must be swiftjepa")
    for label, config in (("baseline", baseline_config), ("engine", engine_config)):
        for key in ("quant", "quantization"):
            if config.get(key) is not None and config.get(key) != "none":
                reasons.append(f"{label} {key} must be disabled for the lossless comparison")

    # Session and input settings must match even when both runs have good timings.
    for key in ("session_id", "seed", "cpu_threads", "opencv_threads"):
        baseline_value, engine_value = baseline_config.get(key), engine_config.get(key)
        if key == "session_id":
            valid = _known_text(baseline_value) and _known_text(engine_value)
        else:
            minimum = 0 if key == "seed" else 1
            valid = all(isinstance(value, int) and not isinstance(value, bool) and value >= minimum
                        for value in (baseline_value, engine_value))
        if not valid or baseline_value != engine_value:
            reasons.append(f"runs must have the same available {key}")
    for key in ("metric", "preprocess"):
        baseline_value, engine_value = baseline_summary.get(key), engine_summary.get(key)
        if not _known_text(baseline_value) or not _known_text(engine_value) or baseline_value != engine_value:
            reasons.append(f"runs must use the same documented {key} semantics")
    metric = baseline_summary.get("metric", "")
    if not isinstance(metric, str) or not all(term in metric for term in ("raw", "uint8", "BTHWC", "embeddings", "CUDA events")):
        reasons.append("latency metric must cover raw uint8 input through embeddings with CUDA events")
    preprocess = baseline_summary.get("preprocess", "")
    if not isinstance(preprocess, str) or not all(term in preprocess for term in ("resize", "center crop", "normalization")):
        reasons.append("preprocessing must document official resize crop and normalization semantics")
    baseline_input = _mapping(baseline_summary.get("input"))
    engine_input = _mapping(engine_summary.get("input"))
    if not baseline_input or baseline_input != engine_input:
        reasons.append("runs must have identical raw input metadata")
    for label, descriptor in (("baseline", baseline_input), ("engine", engine_input)):
        expected_input = {"layout": "BTHWC", "dtype": "uint8", "memory": "pageable CPU",
                          "shape": [1, 16, 256, 256, 3], "crop_size": 256, "short_side_resize": 292}
        for key, expected in expected_input.items():
            value = descriptor.get(key)
            integer_valid = not isinstance(expected, int) or isinstance(value, int) and not isinstance(value, bool)
            shape_valid = key != "shape" or isinstance(value, list) and all(
                isinstance(size, int) and not isinstance(size, bool) for size in value
            )
            if value != expected or not integer_valid or not shape_valid:
                reasons.append(f"{label} raw input {key} must match the target descriptor")
        digest = descriptor.get("sha256")
        if not _hex_digest(digest, 64):
            reasons.append(f"{label} raw timed clip must have a SHA256 digest")

    # Require the same physical target GPU and software stack rather than a model name alone.
    baseline_gpu = _mapping(baseline_environment.get("gpu"))
    engine_gpu = _mapping(engine_environment.get("gpu"))
    for label, gpu in (("baseline", baseline_gpu), ("engine", engine_gpu)):
        name = gpu.get("name", "")
        if not isinstance(name, str) or "RTX 4090" not in name.upper():
            reasons.append(f"{label} GPU must be an RTX 4090")
    baseline_uuid, engine_uuid = baseline_gpu.get("uuid"), engine_gpu.get("uuid")
    if not _known_text(baseline_uuid) or not _known_text(engine_uuid) or baseline_uuid != engine_uuid:
        reasons.append("runs must use the same available GPU UUID")
    version_pairs = (
        ("torch", _mapping(baseline_environment.get("packages")).get("torch"),
         _mapping(engine_environment.get("packages")).get("torch")),
        ("CUDA", baseline_environment.get("torch_cuda_version"), engine_environment.get("torch_cuda_version")),
    )
    for label, baseline_version, engine_version in version_pairs:
        if not _known_text(baseline_version) or not _known_text(engine_version) or baseline_version != engine_version:
            reasons.append(f"runs must use the same available {label} version")
    for label, environment in (("baseline", baseline_environment), ("engine", engine_environment)):
        if environment.get("tracked_dirty") is not False:
            reasons.append(f"{label} tracked source must be clean")
        if not _hex_digest(environment.get("commit"), 40):
            reasons.append(f"{label} source must have a full commit hash")
    if baseline_environment.get("commit") != engine_environment.get("commit"):
        reasons.append("runs must use the same source commit")

    # Reject invalid or noisy runs even if their medians would meet the performance target.
    stages = []
    for label, summary in (("baseline", baseline_summary), ("engine", engine_summary)):
        validity = _mapping(summary.get("validity"))
        stage = _mapping(_mapping(summary.get("stages")).get("end_to_end_ms"))
        stages.append(stage)
        if validity.get("passed") is not True or validity.get("reasons"):
            reasons.append(f"{label} latency validity must pass")
        ratio = _number(validity.get("p95_mean_ratio"), positive=True)
        mean, p95 = _number(stage.get("mean_ms"), positive=True), _number(stage.get("p95_ms"), positive=True)
        computed_ratio = p95 / mean if mean is not None and p95 is not None else None
        if mean is None or p95 is None or ratio is None or computed_ratio is None or not math.isfinite(computed_ratio):
            reasons.append(f"{label} requires finite mean p95 and tail ratio metadata")
        elif not math.isclose(ratio, computed_ratio, rel_tol=1e-9, abs_tol=0):
            reasons.append(f"{label} tail ratio must match its recorded mean and p95")
        if (ratio is not None and ratio > 1.3) or (computed_ratio is not None and computed_ratio > 1.3):
            reasons.append(f"{label} p95 to mean ratio exceeds 1.3")
        count = stage.get("count")
        if count is None:
            count = validity.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 200:
            reasons.append(f"{label} requires at least 200 measured iterations")

    # A failed or skipped baseline quality gate does not remove its required latency comparator.
    baseline_clips = baseline_accuracy.get("clips")
    engine_clips = engine_accuracy.get("clips")
    if isinstance(baseline_clips, list) and baseline_clips:
        baseline_names = [_mapping(row).get("name") for row in baseline_clips]
        engine_names = [_mapping(row).get("name") for row in engine_clips] if isinstance(engine_clips, list) else []
        if sorted(str(name) for name in baseline_names) != sorted(str(name) for name in engine_names):
            reasons.append("runs must validate the same named clip suite")
        for key in ("clip_seed", "clip_layout", "clip_suite_version"):
            if key in baseline_accuracy and baseline_accuracy[key] != engine_accuracy.get(key):
                reasons.append(f"runs must validate the same accuracy {key}")

    measurements = _measurements(stages[0], stages[1], reasons)
    accuracy_reasons = _accuracy_reasons(engine_accuracy, engine_config)
    target_reasons = []
    speedup = measurements["speedup"]
    engine_median = measurements["engine_median_ms"]
    if speedup is None or speedup < 1.5:
        target_reasons.append("engine speedup must be at least 1.5")
    if engine_median is None or engine_median > 12:
        target_reasons.append("engine end to end median must be at most 12 ms")
    return {
        "passed": not reasons and not accuracy_reasons and not target_reasons,
        "comparable": not reasons,
        "accuracy_passed": not accuracy_reasons,
        "performance_targets_met": not target_reasons,
        "speedup": speedup,
        "measurements": measurements,
        "reasons": [*reasons, *accuracy_reasons, *target_reasons],
        "incompatibility_reasons": reasons,
        "accuracy_reasons": accuracy_reasons,
        "target_reasons": target_reasons,
        "targets": {"minimum_speedup": 1.5, "maximum_engine_median_ms": 12.0,
                    "minimum_cosine": 0.999, "model": "vitl", "frames": 16,
                    "batch": 1, "crop_size": 256, "gpu": "RTX 4090"},
    }
