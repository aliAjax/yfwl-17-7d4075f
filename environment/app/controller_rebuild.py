#!/usr/bin/env python3
"""Legacy controller reconstruction helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


REFUSED = "CONTROL-CASE-REFUSED: "


def compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--ledger")
    parser.add_argument("--report")
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def load_case(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported schema")
    return payload


def build_placeholder(case: dict) -> dict:
    channels = list(case.get("channels", []))
    state = [[0.0, 0.0], [0.0, 0.0]]
    input_gain = [[0.0 for _ in channels] for _ in range(2)]
    channel_map = {name: -1 for name in channels}
    result = {
        "case_id": case.get("case_id", ""),
        "closed_loop_gain": [0.0, 0.0],
        "input_channels": channel_map,
        "input_gain": input_gain,
        "poles": [{"imag": 0.0, "real": 0.0}, {"imag": 0.0, "real": 0.0}],
        "schema_version": 1,
        "spectral_radius": 1.0,
        "state_matrix": state,
    }
    digest = hashlib.sha256(compact(result)).hexdigest()
    result["checksum"] = digest
    result["ledger_entry"] = {
        "case_id": result["case_id"],
        "input_sha256": hashlib.sha256(compact(case)).hexdigest(),
        "report_sha256": digest,
        "status": "completed",
    }
    return result


def main() -> int:
    args = parse_args()
    if args.inspect:
        return 0
    if not args.case or not args.ledger or not args.report:
        print(REFUSED + "case ledger and report are required", file=sys.stderr)
        return 3
    try:
        case = load_case(Path(args.case))
        result = build_placeholder(case)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(compact(result))
        ledger_path = Path(args.ledger)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result["ledger_entry"], sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(REFUSED + str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
