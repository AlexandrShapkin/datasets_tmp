#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

import build_cucumber_mixer_open_set_v13 as v13

builder = v13.builder


def is_valid_image(path: Path) -> bool:
    """Reject macOS metadata files and unreadable image payloads."""
    if "__MACOSX" in path.parts or path.name.startswith("._"):
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except Exception:
        return False


def choose_other_images(work: Path, per_class: int = 3) -> list[tuple[str, Path]]:
    root, _ = builder.ensure_snacks(work)
    chosen: list[tuple[str, Path]] = []

    for class_index, class_name in enumerate(builder.OTHER_CLASSES):
        files: set[Path] = set()
        for split in ("train", "val", "test"):
            for extension in ("jpg", "jpeg", "png", "webp"):
                files.update(
                    path.resolve()
                    for path in root.glob(
                        f"**/{split}/{class_name}/*.{extension}"
                    )
                    if path.is_file()
                )

        ordered = sorted(files)
        random.Random(builder.SEED + class_index * 101).shuffle(ordered)

        valid: list[Path] = []
        for path in ordered:
            if is_valid_image(path):
                valid.append(path)
                if len(valid) == per_class:
                    break

        if len(valid) < per_class:
            raise RuntimeError(
                f"Other class {class_name!r} has only {len(valid)} valid images; "
                f"need {per_class}"
            )

        chosen.extend((class_name, path) for path in valid)
        print(
            f"[other] {class_name}: selected={len(valid)} "
            f"from_candidates={len(ordered)}",
            flush=True,
        )

    return chosen


builder.choose_other_images = choose_other_images


def main() -> int:
    return v13.main()


if __name__ == "__main__":
    raise SystemExit(main())
