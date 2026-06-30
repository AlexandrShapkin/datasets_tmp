#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

import build_cucumber_mixer_open_set_v5 as v5

builder = v5.builder
base = builder.base


def main() -> int:
    random.seed(builder.SEED)
    np.random.seed(builder.SEED)
    torch.manual_seed(builder.SEED)

    work = Path(".work_open_set")
    dist = Path("dist_open_set")
    if work.exists():
        shutil.rmtree(work)
    if dist.exists():
        shutil.rmtree(dist)
    work.mkdir(parents=True)
    dist.mkdir(parents=True)

    downloaded_by_class: dict[str, list[base.Candidate]] = {}
    for class_name in builder.TARGETS:
        collected = builder.collect_target_candidates(class_name)
        downloaded = base.download_candidates(
            collected,
            work / "downloads",
            max_downloads=min(1000, len(collected)),
            workers=20,
        )
        unique, rejected = base.preclip_deduplicate(downloaded)
        print(
            f"[dedup] {class_name}: downloaded={len(downloaded)}, "
            f"unique={len(unique)}, rejected={len(rejected)}",
            flush=True,
        )
        if len(unique) < 100:
            raise RuntimeError(
                f"{class_name}: only {len(unique)} unique downloads; need 100"
            )
        downloaded_by_class[class_name] = unique

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(builder.MODEL_NAME)
    model = CLIPModel.from_pretrained(builder.MODEL_NAME).eval().to(device)

    selected: dict[str, list[base.Candidate]] = {}
    for class_name in builder.TARGETS:
        scored = builder.clip_score(
            downloaded_by_class[class_name],
            class_name,
            model,
            processor,
            device,
        )
        selected[class_name] = builder.select_target(scored, 100)
        print(f"[select] {class_name}: {len(selected[class_name])}", flush=True)

    dataset_root = dist / "dataset"
    manifest_rows: list[dict[str, Any]] = []
    all_hashes: set[str] = set()
    target_test_items: list[tuple[str, Path, str, str]] = []

    for class_index, class_name in enumerate(builder.TARGETS):
        items = selected[class_name][:]
        random.Random(builder.SEED + class_index * 1009).shuffle(items)
        train_items = items[:70]
        test_items = items[70:100]

        for index, candidate in enumerate(train_items, start=1):
            filename = f"{class_name}_{index:03d}.jpg"
            destination = dataset_root / "train" / class_name / filename
            _, _, digest = builder.save_clean_jpeg(
                Path(candidate.local_path), destination
            )
            if digest in all_hashes:
                raise RuntimeError(f"Duplicate final image: {candidate.local_path}")
            all_hashes.add(digest)
            manifest_rows.append(
                {
                    "relative_path": destination.relative_to(dataset_root).as_posix(),
                    "class": class_name,
                    "split": "train",
                    "source": candidate.source,
                    "landing_url": candidate.landing_url,
                    "license": candidate.license,
                    "sha256": digest,
                }
            )

        for candidate in test_items:
            target_test_items.append(
                (
                    class_name,
                    Path(candidate.local_path),
                    candidate.source,
                    candidate.landing_url,
                )
            )

    # Ten unrelated source classes, three images from each: 30 `other` images.
    other_items = builder.choose_other_images(work, per_class=3)

    private_test_items: list[tuple[str, str, Path, str, str]] = []
    for class_name, path, source, landing in target_test_items:
        private_test_items.append(
            (class_name, class_name, path, source, landing)
        )
    for source_class, path in other_items:
        private_test_items.append(
            (
                "other",
                source_class,
                path,
                "matthijs_snacks",
                builder.SNACKS_PAGE,
            )
        )

    if len(private_test_items) != 90:
        raise RuntimeError(
            f"Expected 90 test images before shuffle, got {len(private_test_items)}"
        )

    random.Random(builder.SEED + 99991).shuffle(private_test_items)
    ground_truth_rows: list[dict[str, str]] = []
    source_rows: list[dict[str, str]] = []

    for index, (label, source_class, source_path, source, landing) in enumerate(
        private_test_items,
        start=1,
    ):
        filename = f"image_{index:04d}.jpg"
        destination = dataset_root / "test" / filename
        _, _, digest = builder.save_clean_jpeg(source_path, destination)
        if digest in all_hashes:
            raise RuntimeError(f"Train/test duplicate: {source_path}")
        all_hashes.add(digest)
        ground_truth_rows.append({"filename": filename, "class": label})
        source_rows.append(
            {
                "filename": filename,
                "class": label,
                "source_subclass": source_class,
                "source": source,
                "landing_url": landing,
                "sha256": digest,
            }
        )

    readme = """# Cucumber vs concrete mixer: open-set classification dataset

The dataset is intended for a conventional CNN and Ultralytics YOLO
Classification.

## Training set

The training directory contains 140 labeled images:

- `train/cucumber`: 70 images
- `train/concrete_mixer`: 70 images

The `other` class is not used for training.

## Test set

The test directory contains 90 unlabeled images in one flat directory:

- 30 cucumber images
- 30 concrete-mixer images
- 30 unrelated images treated as `other`

Test filenames are neutral (`image_0001.jpg`, ...), and the directory does not
contain class subdirectories.

Because the models are trained only on two known classes, `other` must be
implemented through confidence rejection:

```python
if max_probability < threshold:
    predicted_class = "other"
else:
    predicted_class = known_classes[argmax_probability]
```

Keep `ground_truth_test.csv` outside the dataset and use it only after
inference to calculate metrics and type-I/type-II errors.
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8")

    validation = {
        "status": "ok",
        "target_images_per_known_class": 100,
        "train": {"cucumber": 70, "concrete_mixer": 70},
        "unlabeled_test_total": 90,
        "private_test_distribution": {
            "cucumber": 30,
            "concrete_mixer": 30,
            "other": 30,
        },
        "other_used_for_training": False,
        "unique_final_sha256": len(all_hashes),
        "expected_unique_final_sha256": 230,
        "test_labels_exposed_in_dataset": False,
    }
    (dataset_root / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    builder.write_csv(
        dist / "ground_truth_test.csv",
        ground_truth_rows,
        ["filename", "class"],
    )
    builder.write_csv(
        dist / "source_manifest_private.csv",
        manifest_rows + source_rows,
        sorted(set().union(*(row.keys() for row in manifest_rows + source_rows))),
    )

    builder.contact_sheet(
        [("cucumber", Path(item.local_path)) for item in selected["cucumber"]],
        dist / "contact_sheets" / "cucumber_selected.jpg",
        "Cucumber: 100 selected unique photographs",
    )
    builder.contact_sheet(
        [
            ("concrete_mixer", Path(item.local_path))
            for item in selected["concrete_mixer"]
        ],
        dist / "contact_sheets" / "concrete_mixer_selected.jpg",
        "Concrete mixer: 100 selected unique photographs",
    )
    builder.contact_sheet(
        [(label, path) for label, _, path, _, _ in private_test_items],
        dist / "contact_sheets" / "test_private.jpg",
        "Private test review: 30 cucumber, 30 mixer, 30 other",
    )

    archive_path = dist / "cucumber_concrete_mixer_train70_test30_other30.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(dataset_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dist).as_posix())

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        names = archive.namelist()
        train_cucumber = [
            name
            for name in names
            if name.startswith("dataset/train/cucumber/") and name.endswith(".jpg")
        ]
        train_mixer = [
            name
            for name in names
            if name.startswith("dataset/train/concrete_mixer/") and name.endswith(".jpg")
        ]
        test_names = [
            name
            for name in names
            if name.startswith("dataset/test/") and name.endswith(".jpg")
        ]
        if len(train_cucumber) != 70:
            raise RuntimeError(f"Expected 70 train cucumber images, got {len(train_cucumber)}")
        if len(train_mixer) != 70:
            raise RuntimeError(f"Expected 70 train mixer images, got {len(train_mixer)}")
        if len(test_names) != 90:
            raise RuntimeError(f"Expected 90 test images, got {len(test_names)}")
        if any(
            token in Path(name).name.lower()
            for name in test_names
            for token in ("cucumber", "mixer", "other")
        ):
            raise RuntimeError("A test filename leaks its label")

    summary = {
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        **validation,
    }
    (dist / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
