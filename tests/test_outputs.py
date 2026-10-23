"""Behavior tests for the discrete control handover task."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ENTRY = os.environ.get("CONTROL_ENTRY", "/app/controller_rebuild.py")
REFUSED = "CONTROL-CASE-REFUSED: "
A_TRUE = [[0.72, 0.15], [-0.08, 0.64]]
B_TRUE = [[0.08, 0.05, 0.0], [0.02, 1.0, 0.0]]
CHANNELS = ["P-alpha", "P-gamma", "P-omega"]


def build_payload(sample_count: int = 12, case_id: str = "CTRL-NIGHT-17A") -> dict:
    controls = [
        {
            "P-alpha": round(0.65 * math.sin(index * 1.7) + 0.2 * math.cos(index * 0.4), 10),
            "P-gamma": round(0.7 * math.cos(index * 0.9) + 0.15 * math.sin(index * 2.1), 10),
            "P-omega": round(0.5 * math.sin(index * 2.3 + 0.6), 10),
        }
        for index in range(sample_count - 1)
    ]
    state = [0.4, -0.2]
    samples = [{"controls": controls[0], "state": [0.4, -0.2], "time": 0}]
    for index in range(sample_count - 1):
        control = samples[-1]["controls"]
        next_state = [
            A_TRUE[0][0] * state[0]
            + A_TRUE[0][1] * state[1]
            + B_TRUE[0][0] * control["P-alpha"]
            + B_TRUE[0][1] * control["P-gamma"]
            + B_TRUE[0][2] * control["P-omega"],
            A_TRUE[1][0] * state[0]
            + A_TRUE[1][1] * state[1]
            + B_TRUE[1][0] * control["P-alpha"]
            + B_TRUE[1][1] * control["P-gamma"]
            + B_TRUE[1][2] * control["P-omega"],
        ]
        state = next_state
        next_control = controls[min(index + 1, len(controls) - 1)]
        samples.append(
            {
                "controls": next_control,
                "state": [round(state[0], 12), round(state[1], 12)],
                "time": index + 1,
            }
        )
    return {
        "case_id": case_id,
        "channels": list(CHANNELS),
        "desired_poles": [0.25, 0.45],
        "samples": samples,
        "schema_version": 1,
        "unit": "s",
    }


def write_case(tmp_path: Path, payload: dict, name: str = "case") -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def invoke(
    tmp_path: Path,
    payload: dict | None = None,
    *,
    raw: str | None = None,
    ledger_name: str = "ledger.jsonl",
    report_name: str = "report.json",
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    case_path = tmp_path / "input" / "case.json"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        case_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    else:
        case_path.write_text(raw, encoding="utf-8")
    ledger_path = tmp_path / "state" / ledger_name
    report_path = tmp_path / "out" / report_name
    process = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--case",
            str(case_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    return process, case_path, ledger_path, report_path


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def test_existing_inspect_mode_returns_zero() -> None:
    """题面：「现成的 `--inspect` 模式还应继续以 0 退出」"""
    process = subprocess.run(
        [sys.executable, ENTRY, "--inspect"],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert process.returncode == 0

def test_success_is_silent_and_creates_report(tmp_path: Path) -> None:
    """题面：「正常完成时终端保持安静，标准输出与标准错误均为空」"""
    process, _, _, report = invoke(tmp_path, build_payload())
    assert process.returncode == 0
    assert process.stdout == ""
    assert process.stderr == ""
    assert report.is_file()

def test_missing_argument_returns_three(tmp_path: Path) -> None:
    """题面：「拒绝路径返回状态码 3」"""
    report_path = tmp_path / "out" / "case_02.json"
    process = subprocess.run(
        [sys.executable, ENTRY, "--case", str(tmp_path / "case_00.json")],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stdout == ""
    assert process.stderr.startswith(REFUSED)

def test_missing_case_returns_three(tmp_path: Path) -> None:
    """题面：「拒绝路径返回状态码 3」"""
    case_path = tmp_path / "input" / "case.json"
    ledger_path = tmp_path / "state" / "ledger.jsonl"
    report_path = tmp_path / "out" / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--case",
            str(case_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert process.returncode == 3
    assert report_path.exists() is False
    assert ledger_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_malformed_json_returns_three(tmp_path: Path) -> None:
    """题面：「读入对象由下列字段构成」"""
    process, _, _, report_path = invoke(tmp_path, raw="{not-json")
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stdout == ""
    assert process.stderr.startswith(REFUSED)

def test_duplicate_top_level_key_returns_three(tmp_path: Path) -> None:
    """题面：「读入对象由下列字段构成：」"""
    raw = (
        '{"case_id":"A","case_id":"B","channels":["a","b","c"],'
        '"desired_poles":[0.2,0.4],"samples":[],"schema_version":1,"unit":"s"}'
    )
    process, _, _, report_path = invoke(tmp_path, raw=raw)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_top_level_array_returns_three(tmp_path: Path) -> None:
    """题面：「读入对象由下列字段构成」"""
    process, _, _, report_path = invoke(tmp_path, raw="[]")
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_missing_top_level_key_returns_three(tmp_path: Path) -> None:
    """题面：「读入对象由下列字段构成：」"""
    payload = build_payload()
    del payload["unit"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_extra_top_level_key_returns_three(tmp_path: Path) -> None:
    """题面：「读入对象由下列字段构成：」"""
    payload = build_payload()
    payload["operator_note"] = "ignore this"
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_schema_version_bool_returns_three(tmp_path: Path) -> None:
    """题面：「| `schema_version` | 整数 1 |」"""
    payload = build_payload()
    payload["schema_version"] = True
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_unit_value_returns_three(tmp_path: Path) -> None:
    """题面：「读入对象由下列字段构成」"""
    payload = build_payload()
    payload["unit"] = "min"
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_empty_case_id_returns_three(tmp_path: Path) -> None:
    """题面：「| `case_id` | 非空 |」"""
    payload = build_payload(case_id="")
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_duplicate_channel_returns_three(tmp_path: Path) -> None:
    """题面：「| `channels` | 三项互不重复的端口名 |」"""
    payload = build_payload()
    payload["channels"] = ["P-alpha", "P-alpha", "P-omega"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_desired_pole_boundary_returns_three(tmp_path: Path) -> None:
    """题面：「| `desired_poles` | 两个绝对值小于 1 的数 |」"""
    payload = build_payload()
    payload["desired_poles"] = [1.0, 0.4]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_too_few_samples_returns_three(tmp_path: Path) -> None:
    """题面：「| `samples` | 8 至 64 条采样 |」"""
    process, _, _, report_path = invoke(tmp_path, build_payload(sample_count=7))
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_too_many_samples_returns_three(tmp_path: Path) -> None:
    """题面：「| `samples` | 8 至 64 条采样 |」"""
    process, _, _, report_path = invoke(tmp_path, build_payload(sample_count=65))
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_first_sample_time_nonzero_returns_three(tmp_path: Path) -> None:
    """题面：「| 采样时间 | 从 0 起按整数递增 |」"""
    payload = build_payload()
    payload["samples"][0]["time"] = 1
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_non_monotonic_time_returns_three(tmp_path: Path) -> None:
    """题面：「| 采样时间 | 从 0 起按整数递增 |」"""
    payload = build_payload()
    payload["samples"][3]["time"] = 2
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_state_type_returns_three(tmp_path: Path) -> None:
    """题面：「| 状态向量 | 两个有限数 |」"""
    payload = build_payload()
    payload["samples"][2]["state"][1] = True
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_control_unknown_channel_returns_three(tmp_path: Path) -> None:
    """题面：「| 控制字典 | 键与 `channels` 对齐，值为有限数 |」"""
    payload = build_payload()
    payload["samples"][4]["controls"]["P-extra"] = 1.0
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_control_missing_channel_returns_three(tmp_path: Path) -> None:
    """题面：「| 控制字典 | 键与 `channels` 对齐，值为有限数 |」"""
    payload = build_payload()
    del payload["samples"][4]["controls"]["P-omega"]
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_control_bool_returns_three(tmp_path: Path) -> None:
    """题面：「| 控制字典 | 键与 `channels` 对齐，值为有限数 |」"""
    payload = build_payload()
    payload["samples"][4]["controls"]["P-gamma"] = False
    process, _, _, report_path = invoke(tmp_path, payload)
    assert process.returncode == 3
    assert report_path.exists() is False
    assert process.stderr.startswith(REFUSED)

def test_report_exact_keys(tmp_path: Path) -> None:
    """题面：「交付 JSON 的字段表如下」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    report = read_report(report_path)
    assert process.returncode == 0
    assert set(report) == {
        "case_id",
        "checksum",
        "closed_loop_gain",
        "input_channels",
        "input_gain",
        "ledger_entry",
        "poles",
        "schema_version",
        "spectral_radius",
        "state_matrix",
    }

def test_report_state_matrix_recovered(tmp_path: Path) -> None:
    """题面：「正确实现要从状态转移里恢复两行状态矩阵和两行输入增益」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    matrix = read_report(report_path)["state_matrix"]
    assert process.returncode == 0
    assert abs(matrix[0][0] - A_TRUE[0][0]) < 1e-8
    assert abs(matrix[0][1] - A_TRUE[0][1]) < 1e-8
    assert abs(matrix[1][0] - A_TRUE[1][0]) < 1e-8
    assert abs(matrix[1][1] - A_TRUE[1][1]) < 1e-8

def test_report_input_gain_recovered(tmp_path: Path) -> None:
    """题面：「正确实现要从状态转移里恢复两行状态矩阵和两行输入增益」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    gain = read_report(report_path)["input_gain"]
    assert process.returncode == 0
    assert abs(gain[0][0] - B_TRUE[0][0]) < 1e-8
    assert abs(gain[0][1] - B_TRUE[0][1]) < 1e-8
    assert abs(gain[1][1] - B_TRUE[1][1]) < 1e-8

def test_report_input_channel_mapping(tmp_path: Path) -> None:
    """题面：「`input_channels` 的键恰好覆盖 `channels`，值为该端口主导影响的状态分量下标」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    mapping = read_report(report_path)["input_channels"]
    assert process.returncode == 0
    assert mapping == {"P-alpha": 0, "P-gamma": 1, "P-omega": -1}

def test_report_closed_loop_gain_stable(tmp_path: Path) -> None:
    """题面：「`closed_loop_gain` 是按状态顺序排列的一行反馈增益」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    gain = read_report(report_path)["closed_loop_gain"]
    assert process.returncode == 0
    assert len(gain) == 2
    assert all(math.isfinite(value) for value in gain)
    assert abs(gain[0]) > 0.0

def test_report_poles_match_requirement(tmp_path: Path) -> None:
    """题面：「`poles` 恰好两个对象」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    poles = read_report(report_path)["poles"]
    real_parts = sorted(item["real"] for item in poles)
    assert process.returncode == 0
    assert real_parts == pytest.approx([0.25, 0.45], abs=1e-6)

def test_report_spectral_radius_under_one(tmp_path: Path) -> None:
    """题面：「`spectral_radius` 是闭环极点模长的最大值」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    radius = read_report(report_path)["spectral_radius"]
    assert process.returncode == 0
    assert radius < 1.0
    assert radius == pytest.approx(0.45, abs=1e-6)

def test_report_compact_trailing_newline(tmp_path: Path) -> None:
    """题面：「报告必须使用按键字典序排列、无空格的紧凑 JSON，且以单个换行结束」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    raw = report_path.read_bytes()
    parsed = json.loads(raw)
    assert process.returncode == 0
    assert raw == compact(parsed)
    assert raw.endswith(b"\n")

def test_checksum_matches_result(tmp_path: Path) -> None:
    """题面：「`checksum` 是恢复结果与闭环结果紧凑 JSON 的 SHA-256 十六进制摘要」"""
    process, _, _, report_path = invoke(tmp_path, build_payload())
    report = read_report(report_path)
    base = {
        key: value
        for key, value in report.items()
        if key not in {"checksum", "ledger_entry"}
    }
    expected = hashlib.sha256(compact(base)).hexdigest()
    assert process.returncode == 0
    assert report["checksum"] == expected

def test_ledger_first_run_appends_one(tmp_path: Path) -> None:
    """题面：「成功记录写入运行台账 JSONL」"""
    process, _, ledger_path, _ = invoke(tmp_path, build_payload())
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert process.returncode == 0
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "completed"

def test_ledger_repeat_run_is_idempotent(tmp_path: Path) -> None:
    """题面：「同一案例、同一输入摘要与同一报告字节再次提交时，流水不新增行，报告仍逐字节一致」"""
    first, case_path, ledger_path, report_path = invoke(tmp_path, build_payload())
    before = report_path.read_bytes()
    second = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--case",
            str(case_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert first.returncode == 0
    assert second.returncode == 0
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 1
    assert report_path.read_bytes() == before

def test_ledger_changed_input_appends(tmp_path: Path) -> None:
    """题面：「新输入摘要才追加记录」"""
    first, case_path, ledger_path, _ = invoke(tmp_path, build_payload())
    changed = build_payload(case_id="CTRL-NIGHT-17B")
    case_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    second = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--case",
            str(case_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(tmp_path / "out" / "report-two.json"),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert first.returncode == 0
    assert second.returncode == 0
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 2

def test_failure_preserves_existing_outputs(tmp_path: Path) -> None:
    """题面：「输入校验失败、拟合退化或闭环不可镇定时不修改旧报告和运行台账，也不留下临时文件」"""
    success, case_path, ledger_path, report_path = invoke(tmp_path, build_payload())
    ledger_before = ledger_path.read_bytes()
    report_before = report_path.read_bytes()
    case_path.write_text("{broken", encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--case",
            str(case_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert success.returncode == 0
    assert failed.returncode == 3
    assert ledger_path.read_bytes() == ledger_before
    assert report_path.read_bytes() == report_before

def test_success_does_not_mutate_case(tmp_path: Path) -> None:
    """题面：「每个采样时刻的 `controls` 是保持到下一采样时刻的端口输入」"""
    process, case_path, _, _ = invoke(tmp_path, build_payload())
    before = write_case(tmp_path, build_payload(), name="expected")
    assert process.returncode == 0
    assert case_path.read_bytes() == before.read_bytes()

def test_report_bytes_stable_across_ledger_paths(tmp_path: Path) -> None:
    """题面：「报告仍逐字节一致」"""
    first, _, _, report_one = invoke(tmp_path, build_payload(), ledger_name="one.jsonl", report_name="one.json")
    second, _, _, report_two = invoke(tmp_path, build_payload(), ledger_name="two.jsonl", report_name="two.json")
    assert first.returncode == 0
    assert second.returncode == 0
    assert report_one.read_bytes() == report_two.read_bytes()

def test_maximum_sample_count_accepted(tmp_path: Path) -> None:
    """题面：「| `samples` | 8 至 64 条采样 |」"""
    process, _, _, report_path = invoke(tmp_path, build_payload(sample_count=64))
    assert process.returncode == 0
    assert report_path.is_file()

def test_minimum_sample_count_accepted(tmp_path: Path) -> None:
    """题面：「| `samples` | 8 至 64 条采样 |」"""
    process, _, _, report_path = invoke(tmp_path, build_payload(sample_count=8))
    assert process.returncode == 0
    assert report_path.is_file()

def test_no_temp_files_on_success(tmp_path: Path) -> None:
    """题面：「输入校验失败、拟合退化或闭环不可镇定时不修改旧报告和运行台账，也不留下临时文件」"""
    process, _, ledger_path, report_path = invoke(tmp_path, build_payload())
    leftovers = [
        item.name
        for item in [*report_path.parent.iterdir(), *ledger_path.parent.iterdir()]
        if item.name.startswith(".")
    ]
    assert process.returncode == 0
    assert leftovers == []

def test_normal_case_finishes_within_three_seconds(tmp_path: Path) -> None:
    """题面：「正常规模的单次处理不应超过 3 秒」"""
    case_path = write_case(tmp_path, build_payload())
    ledger_path = tmp_path / "state" / "timed.jsonl"
    report_path = tmp_path / "out" / "timed.json"
    start = time.monotonic()
    process = subprocess.run(
        [
            sys.executable,
            ENTRY,
            "--case",
            str(case_path),
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    elapsed = time.monotonic() - start
    assert process.returncode == 0
    assert report_path.is_file()
    assert elapsed < 3.0
