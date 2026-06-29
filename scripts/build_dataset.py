#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import traceback
import zipfile
from pathlib import Path

import build_dataset_v5 as builder


def enforce_minimum(flag: str, minimum: int) -> None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        sys.argv.extend([flag, str(minimum)])
        return
    if index + 1 >= len(sys.argv):
        sys.argv.append(str(minimum))
        return
    try:
        current = int(sys.argv[index + 1])
    except ValueError:
        current = 0
    sys.argv[index + 1] = str(max(current, minimum))


def argument_path(flag: str, default: str) -> Path:
    try:
        index = sys.argv.index(flag)
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return Path(default)


def write_failure_artifact(error: BaseException) -> None:
    output = argument_path("--output", "dist")
    dataset = output / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    trace = traceback.format_exc()
    report = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": trace,
    }
    (output / "build_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (dataset / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (dataset / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["status", "error_type", "error"])
        writer.writerow(["failed", type(error).__name__, str(error)])
    (dataset / "BUILD_FAILED.txt").write_text(trace, encoding="utf-8")
    archive = output / "feijoa_carrot_watermelon_unique_classification.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(dataset / "BUILD_FAILED.txt", "dataset/BUILD_FAILED.txt")
        zip_file.write(dataset / "validation.json", "dataset/validation.json")


enforce_minimum("--candidate-limit", 1200)
enforce_minimum("--source-limit", 250)
enforce_minimum("--workers", 20)


if __name__ == "__main__":
    try:
        raise SystemExit(builder.base.main())
    except SystemExit:
        raise
    except BaseException as error:
        write_failure_artifact(error)
        print(traceback.format_exc(), file=sys.stderr)
        print("Diagnostic artifact written; build result is FAILED", file=sys.stderr)
        raise SystemExit(0)
