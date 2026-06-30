#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import traceback
from pathlib import Path

OUTPUT = Path("dist/kaggle_probe.txt")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
lines: list[str] = []


def log(value: object) -> None:
    text = str(value)
    print(text, flush=True)
    lines.append(text)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


log("Kaggle Fruits-262 probe")
log(f"KAGGLE_API_TOKEN present: {bool(os.environ.get('KAGGLE_API_TOKEN'))}")
log(f"KAGGLE_USERNAME present: {bool(os.environ.get('KAGGLE_USERNAME'))}")

try:
    command = [
        "kaggle",
        "datasets",
        "files",
        "aelchimminut/fruits262",
        "--csv",
    ]
    process = subprocess.run(command, text=True, capture_output=True, timeout=120)
    log(f"CLI returncode={process.returncode}")
    log("CLI STDOUT")
    log(process.stdout[:100000])
    log("CLI STDERR")
    log(process.stderr[:20000])
except BaseException as exc:
    log(f"CLI failed: {type(exc).__name__}: {exc}")
    log(traceback.format_exc())

try:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
        log("kaggle-api authenticate: success")
    except BaseException as exc:
        log(f"kaggle-api authenticate: {type(exc).__name__}: {exc}")

    try:
        response = api.dataset_list_files("aelchimminut/fruits262")
        files = list(getattr(response, "files", []) or [])
        log(f"dataset_list_files count first page: {len(files)}")
        for item in files:
            name = str(getattr(item, "name", ""))
            if "feijoa" in name.lower():
                log(f"FEIJOA_FILE {name} size={getattr(item, 'total_bytes', '')}")
        log(f"next_page_token={getattr(response, 'next_page_token', None)}")
        if files:
            log("FIRST_FILES")
            for item in files[:50]:
                log(str(getattr(item, "name", item)))
    except BaseException as exc:
        log(f"dataset_list_files failed: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
except BaseException as exc:
    log(f"kaggle package failed: {type(exc).__name__}: {exc}")
    log(traceback.format_exc())

try:
    import kagglehub

    log(f"kagglehub version={getattr(kagglehub, '__version__', 'unknown')}")
    guesses = [
        "Fruit-262/feijoa/0.jpg",
        "Fruit-262/feijoa/1.jpg",
        "Fruit-262/feijoa/0000.jpg",
        "Fruit-262/feijoa/0001.jpg",
        "Resized_Fruits-262/feijoa/0.jpg",
        "Resized_Fruits-262/feijoa/1.jpg",
        "Resized_Fruits-262/feijoa/0000.jpg",
        "Resized_Fruits-262/feijoa/0001.jpg",
    ]
    for guess in guesses:
        try:
            path = kagglehub.dataset_download(
                "aelchimminut/fruits262",
                path=guess,
                output_dir="dist/kaggle_single",
            )
            log(f"KAGGLEHUB_SUCCESS {guess} -> {path}")
            break
        except BaseException as exc:
            log(f"KAGGLEHUB_FAIL {guess}: {type(exc).__name__}: {exc}")
except BaseException as exc:
    log(f"kagglehub failed: {type(exc).__name__}: {exc}")
    log(traceback.format_exc())

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
