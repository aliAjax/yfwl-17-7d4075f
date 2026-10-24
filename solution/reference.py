#!/usr/bin/env python3
"""Recover a discrete two-state model and publish a stabilizing controller.

All matrix arithmetic uses double-precision floating point so the recovered
state-space coefficients, poles, and feedback gain share one numerical basis.
数值精度统一采用双精度浮点，避免闭环极点与反馈增益落在不同取整口径。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REFUSED = "CONTROL-CASE-REFUSED: "
TOP_KEYS = {
    "case_id",
    "channels",
    "desired_poles",
    "samples",
    "schema_version",
    "unit",
}
SAMPLE_KEYS = {"controls", "state", "time"}
EPSILON = 1e-9


class Refusal(ValueError):
    """The scenario cannot be admitted."""


def finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(f"duplicate key: {key}")
        result[key] = value
    return result


def load_scenario(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"cannot read scenario: {exc}") from exc
    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise Refusal(f"invalid scenario JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Refusal("scenario root must be an object")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Refusal(f"{label} keys differ; missing={missing} extra={extra}")


def validate_scenario(raw: dict[str, Any]) -> dict[str, Any]:
    require_exact_keys(raw, TOP_KEYS, "scenario")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise Refusal("schema_version must be integer 1")
    if raw["unit"] != "s":
        raise Refusal("unit must be s")
    case_id = raw["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise Refusal("case_id must be a non-empty string")

    channels = raw["channels"]
    if not isinstance(channels, list) or len(channels) != 3:
        raise Refusal("channels must contain exactly three entries")
    if any(not isinstance(item, str) or not item.strip() for item in channels):
        raise Refusal("channel names must be non-empty strings")
    if len(set(channels)) != len(channels):
        raise Refusal("channel names must be unique")

    poles = raw["desired_poles"]
    if not isinstance(poles, list) or len(poles) != 2:
        raise Refusal("desired_poles must contain two entries")
    if any(not finite_number(item) or abs(float(item)) >= 1.0 for item in poles):
        raise Refusal("desired poles must be finite and inside the unit circle")

    samples = raw["samples"]
    if not isinstance(samples, list) or not 8 <= len(samples) <= 64:
        raise Refusal("samples must contain between 8 and 64 records")

    normalized_samples: list[dict[str, Any]] = []
    expected_channels = set(channels)
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise Refusal(f"samples[{index}] must be an object")
        require_exact_keys(sample, SAMPLE_KEYS, f"samples[{index}]")
        timestamp = sample["time"]
        if type(timestamp) is not int or timestamp != index:
            raise Refusal(
                f"samples[{index}].time must equal its zero-based sample index"
            )

        state = sample["state"]
        if (
            not isinstance(state, list)
            or len(state) != 2
            or any(not finite_number(item) for item in state)
        ):
            raise Refusal(f"samples[{index}].state must contain two finite numbers")

        controls = sample["controls"]
        if not isinstance(controls, dict) or set(controls) != expected_channels:
            raise Refusal(f"samples[{index}].controls must exactly cover channels")
        if any(not finite_number(value) for value in controls.values()):
            raise Refusal(f"samples[{index}].controls must be finite numbers")

        normalized_samples.append(
            {
                "time": timestamp,
                "state": [float(state[0]), float(state[1])],
                "controls": {name: float(controls[name]) for name in channels},
            }
        )

    return {
        "case_id": case_id,
        "channels": list(channels),
        "desired_poles": [float(item) for item in poles],
        "samples": normalized_samples,
        "schema_version": 1,
        "unit": "s",
    }


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    work = [list(map(float, matrix[row])) + [float(vector[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < EPSILON:
            raise Refusal("regression matrix is singular")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        for cell in range(column, size + 1):
            work[column][cell] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if abs(factor) < EPSILON:
                continue
            for cell in range(column, size + 1):
                work[row][cell] -= factor * work[column][cell]
    return [work[row][size] for row in range(size)]


def least_squares(rows: list[list[float]], targets: dict[int, list[float]]) -> list[list[float]]:
    width = len(rows[0])
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    right = {0: [0.0 for _ in range(width)], 1: [0.0 for _ in range(width)]}
    for row, target in zip(rows, zip(targets[0], targets[1])):
        for left in range(width):
            for other in range(width):
                normal[left][other] += row[left] * row[other]
            right[0][left] += row[left] * target[0]
            right[1][left] += row[left] * target[1]
    first = solve_linear(normal, right[0])
    second = solve_linear(normal, right[1])
    return [first, second]


def fit_model(case: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    channels = case["channels"]
    rows: list[list[float]] = []
    next_first: list[float] = []
    next_second: list[float] = []
    samples = case["samples"]
    for current, following in zip(samples, samples[1:]):
        x0, x1 = current["state"]
        controls = [current["controls"][name] for name in channels]
        rows.append([x0, x1, *controls])
        next_first.append(following["state"][0])
        next_second.append(following["state"][1])
    coefficients = least_squares(rows, {0: next_first, 1: next_second})
    state_matrix = [
        [coefficients[0][0], coefficients[0][1]],
        [coefficients[1][0], coefficients[1][1]],
    ]
    input_gain = [
        [coefficients[0][2], coefficients[0][3], coefficients[0][4]],
        [coefficients[1][2], coefficients[1][3], coefficients[1][4]],
    ]
    for row in state_matrix + input_gain:
        for value in row:
            if not math.isfinite(value):
                raise Refusal("fitted coefficients are not finite")
    return state_matrix, input_gain


def channel_map(input_gain: list[list[float]], channels: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for index, name in enumerate(channels):
        values = [input_gain[0][index], input_gain[1][index]]
        scale = max(abs(values[0]), abs(values[1]))
        if scale < 1e-6:
            mapping[name] = -1
            continue
        dominant = 0 if abs(values[0]) >= abs(values[1]) else 1
        if dominant in used:
            raise Refusal("input channel correspondence is not unique")
        used.add(dominant)
        mapping[name] = dominant
    if len(used) != 2:
        raise Refusal("input channels must cover both state components")
    return mapping


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            math.fsum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def matadd(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def matscale(matrix: list[list[float]], factor: float) -> list[list[float]]:
    return [[factor * value for value in row] for row in matrix]


def identity(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        list(map(float, matrix[row])) + identity(size)[row]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < EPSILON:
            raise Refusal("controllability matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(2 * size)
            ]
    return [row[size:] for row in augmented]


def stabilizer(
    state_matrix: list[list[float]],
    input_gain: list[list[float]],
    mapping: dict[str, int],
    desired_poles: list[float],
) -> list[float]:
    active_names = [name for name, value in mapping.items() if value >= 0]
    active_index = max(
        active_names,
        key=lambda name: math.hypot(
            input_gain[0][list(mapping).index(name)],
            input_gain[1][list(mapping).index(name)],
        ),
    )
    channel_position = list(mapping).index(active_index)
    b = [[input_gain[0][channel_position]], [input_gain[1][channel_position]]]
    a_b = matmul(state_matrix, b)
    controllability = [
        [b[0][0], a_b[0][0]],
        [b[1][0], a_b[1][0]],
    ]
    inverse = invert(controllability)
    a_squared = matmul(state_matrix, state_matrix)
    a_scaled = matscale(state_matrix, -(desired_poles[0] + desired_poles[1]))
    polynomial = matadd(matadd(a_squared, a_scaled), matscale(identity(2), desired_poles[0] * desired_poles[1]))
    selector = [[0.0, 1.0]]
    gain = matmul(matmul(selector, inverse), polynomial)
    return [gain[0][0], gain[0][1]]


def eigenvalues(matrix: list[list[float]]) -> list[dict[str, float]]:
    trace = matrix[0][0] + matrix[1][1]
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    discriminant = trace * trace - 4.0 * determinant
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        return [
            {"real": (trace + root) / 2.0, "imag": 0.0},
            {"real": (trace - root) / 2.0, "imag": 0.0},
        ]
    imag = math.sqrt(-discriminant) / 2.0
    return [
        {"real": trace / 2.0, "imag": imag},
        {"real": trace / 2.0, "imag": -imag},
    ]


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_report(case: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    state_matrix, input_gain = fit_model(case)
    mapping = channel_map(input_gain, case["channels"])
    gain = stabilizer(state_matrix, input_gain, mapping, case["desired_poles"])
    active_names = [name for name, value in mapping.items() if value >= 0]
    chosen = max(
        active_names,
        key=lambda name: math.hypot(
            input_gain[0][list(mapping).index(name)],
            input_gain[1][list(mapping).index(name)],
        ),
    )
    chosen_column = list(mapping).index(chosen)
    feedback = [[gain[0], gain[1]]]
    active_gain = [
        [input_gain[0][chosen_column]],
        [input_gain[1][chosen_column]],
    ]
    closed_loop = matadd(state_matrix, matscale(matmul(active_gain, feedback), -1.0))
    poles = eigenvalues(closed_loop)
    spectral_radius = max(math.hypot(item["real"], item["imag"]) for item in poles)
    if spectral_radius >= 1.0 - 1e-8:
        raise Refusal("closed loop is not stable")

    base = {
        "case_id": case["case_id"],
        "closed_loop_gain": gain,
        "input_channels": {name: mapping[name] for name in sorted(mapping)},
        "input_gain": input_gain,
        "poles": sorted(poles, key=lambda item: (item["real"], item["imag"])),
        "schema_version": 1,
        "spectral_radius": spectral_radius,
        "state_matrix": state_matrix,
    }
    checksum = hashlib.sha256(compact_bytes(base)).hexdigest()
    core = {**base, "checksum": checksum}
    report_digest = hashlib.sha256(compact_bytes(core)).hexdigest()
    entry = {
        "case_id": case["case_id"],
        "input_sha256": hashlib.sha256(compact_bytes(case)).hexdigest(),
        "report_sha256": report_digest,
        "status": "completed",
    }
    return {**core, "ledger_entry": entry}, compact_bytes({**core, "ledger_entry": entry})


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            record = json.loads(line, object_pairs_hook=reject_duplicates)
        except (json.JSONDecodeError, Refusal) as exc:
            raise Refusal(f"ledger line {number} is invalid: {exc}") from exc
        if not isinstance(record, dict):
            raise Refusal(f"ledger line {number} must be an object")
        records.append(record)
    return records


def stage_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return Path(temporary)


def publish(report_path: Path, ledger_path: Path, result: dict[str, Any], report_bytes: bytes) -> None:
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        records = read_ledger(ledger_path)
        existing = next(
            (
                item
                for item in records
                if item.get("case_id") == result["ledger_entry"]["case_id"]
                and item.get("input_sha256") == result["ledger_entry"]["input_sha256"]
                and item.get("report_sha256") == result["ledger_entry"]["report_sha256"]
                and item.get("status") == "completed"
            ),
            None,
        )
        report_stage = stage_write(report_path, report_bytes)
        ledger_stage: Path | None = None
        try:
            if existing is None:
                ledger_payload = b"".join(
                    compact_bytes(item) for item in [*records, result["ledger_entry"]]
                )
                ledger_stage = stage_write(ledger_path, ledger_payload)
                os.replace(ledger_stage, ledger_path)
                ledger_stage = None
            os.replace(report_stage, report_path)
            report_stage = None
        finally:
            if report_stage is not None:
                report_stage.unlink(missing_ok=True)
            if ledger_stage is not None:
                ledger_stage.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--ledger")
    parser.add_argument("--report")
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inspect:
        return 0
    if not args.case or not args.ledger or not args.report:
        print(REFUSED + "case ledger and report are required", file=sys.stderr)
        return 3
    try:
        raw = load_scenario(Path(args.case))
        case = validate_scenario(raw)
        result, report_bytes = build_report(case)
        publish(Path(args.report), Path(args.ledger), result, report_bytes)
        return 0
    except (Refusal, OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
        print(REFUSED + str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
