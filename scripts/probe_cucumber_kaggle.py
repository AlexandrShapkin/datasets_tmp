#!/usr/bin/env python3
from __future__ import annotations

import traceback
from pathlib import Path

import kagglehub

HANDLE = "kritikseth/fruit-and-vegetable-image-recognition"
GUESSES = [
    "train/cucumber/Image_1.jpg",
    "train/cucumber/Image_2.jpg",
    "test/cucumber/Image_1.jpg",
    "validation/cucumber/Image_1.jpg",
    "Fruit and Vegetable Image Recognition/train/cucumber/Image_1.jpg",
    "Fruit and Vegetable Image Recognition/test/cucumber/Image_1.jpg",
    "Fruit and Vegetable Image Recognition/validation/cucumber/Image_1.jpg",
]

lines: list[str] = []
for guess in GUESSES:
    try:
        path = kagglehub.dataset_download(
            HANDLE,
            path=guess,
            output_dir="dist_cucumber_probe/files",
        )
        lines.append(f"SUCCESS {guess} -> {path}")
    except BaseException as exc:
        lines.append(f"FAIL {guess}: {type(exc).__name__}: {exc}")

output = Path("dist_cucumber_probe/report.txt")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(output.read_text(encoding="utf-8"))
