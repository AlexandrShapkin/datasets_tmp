#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

import build_dataset_v5 as v5

base = v5.base

ORIGINAL_CLIP_SCORE = base.clip_score_candidates
ORIGINAL_CONTENT_FILTER = base.content_filter
ORIGINAL_BUILD_README = base.build_readme

FEIJOA_FRUIT_PROMPTS = [
    "a close-up photograph of a green oval feijoa fruit",
    "a photograph of harvested feijoa fruits in a hand",
    "a photograph of several green feijoa fruits",
    "a cut feijoa fruit showing pale edible flesh",
    "green feijoa fruits hanging from a branch",
    "pineapple guava fruit, an oval green edible fruit",
]

FEIJOA_FLOWER_PROMPTS = [
    "a close-up photograph of a red and white feijoa flower",
    "a feijoa blossom with red stamens and white petals",
    "red flowers on a feijoa shrub without visible fruit",
    "a flowering plant with red and white blossoms",
    "a botanical flower close-up without fruit",
]

FEIJOA_NONFRUIT_PROMPTS = [
    "green leaves and branches without visible fruit",
    "a whole shrub where no fruit can be clearly seen",
    "flower buds and leaves without edible fruit",
]

FLOWER_WORDS = {
    "flower",
    "flowers",
    "flowering",
    "blossom",
    "blossoms",
    "bloom",
    "blooms",
    "inflorescence",
    "floral",
}


def feijoa_color_fractions(path: str) -> tuple[float, float, float]:
    """Return green, red/pink and neutral fractions on a small HSV image."""
    with Image.open(path) as image:
        image.load()
        hsv = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128)).convert("HSV")
    array = np.asarray(hsv, dtype=np.uint8)
    hue = array[..., 0]
    saturation = array[..., 1]
    value = array[..., 2]

    colorful = (saturation >= 55) & (value >= 45)
    green = colorful & (hue >= 35) & (hue <= 115)
    red_pink = colorful & ((hue <= 18) | (hue >= 220))
    neutral = saturation < 45
    pixels = float(hue.size)
    return (
        float(green.sum() / pixels),
        float(red_pink.sum() / pixels),
        float(neutral.sum() / pixels),
    )


def clip_score_candidates(
    candidates: list[base.Candidate],
    class_name: str,
    model: base.CLIPModel,
    processor: base.CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> list[base.Candidate]:
    if class_name != "feijoa":
        return ORIGINAL_CLIP_SCORE(
            candidates, class_name, model, processor, device, batch_size
        )

    fruit_features = base.clip_text_features(
        model, processor, FEIJOA_FRUIT_PROMPTS, device
    )
    flower_features = base.clip_text_features(
        model, processor, FEIJOA_FLOWER_PROMPTS, device
    )
    nonfruit_features = base.clip_text_features(
        model, processor, FEIJOA_NONFRUIT_PROMPTS, device
    )

    output: list[base.Candidate] = []
    for chunk in base.batch(candidates, batch_size):
        images: list[Image.Image] = []
        valid: list[base.Candidate] = []
        for candidate in chunk:
            try:
                with Image.open(candidate.local_path) as image:
                    image.load()
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
                    valid.append(candidate)
            except Exception:
                continue
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings = features.detach().cpu().numpy().astype(np.float32)

        fruit_scores = (embeddings @ fruit_features.T).max(axis=1)
        flower_scores = (embeddings @ flower_features.T).max(axis=1)
        nonfruit_scores = (embeddings @ nonfruit_features.T).max(axis=1)

        for candidate, embedding, fruit, flower, nonfruit in zip(
            valid, embeddings, fruit_scores, flower_scores, nonfruit_scores
        ):
            green_fraction, red_fraction, neutral_fraction = feijoa_color_fractions(
                candidate.local_path
            )
            visual_negative = max(float(flower), float(nonfruit))
            raw_margin = float(fruit) - visual_negative
            color_adjustment = min(green_fraction, 0.45) * 0.12
            color_adjustment -= max(0.0, red_fraction - 0.018) * 1.20
            adjusted_margin = raw_margin + color_adjustment

            candidate.clip_positive = float(fruit)
            candidate.clip_negative = visual_negative
            candidate.clip_margin = adjusted_margin
            candidate.clip_embedding = embedding.tolist()
            candidate.feijoa_raw_margin = raw_margin
            candidate.feijoa_green_fraction = green_fraction
            candidate.feijoa_red_fraction = red_fraction
            candidate.feijoa_neutral_fraction = neutral_fraction
            output.append(candidate)

        print(f"[clip-fruit-only] feijoa: scored {len(output)}/{len(candidates)}")
    return output


base.clip_score_candidates = clip_score_candidates


def metadata_contains_flower(candidate: base.Candidate) -> bool:
    text = f"{candidate.title} {candidate.query}".lower()
    words = set(
        text.replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace(",", " ")
        .split()
    )
    return bool(words & FLOWER_WORDS)


def content_filter(
    candidates: list[base.Candidate],
    minimum_needed: int,
) -> tuple[list[base.Candidate], dict[str, Any]]:
    if not candidates or candidates[0].class_name != "feijoa":
        return ORIGINAL_CONTENT_FILTER(candidates, minimum_needed)

    strict: list[base.Candidate] = []
    rejected_reasons: dict[str, int] = {
        "metadata_flower": 0,
        "red_or_pink": 0,
        "insufficient_green": 0,
        "clip_fruit_score": 0,
        "fruit_vs_flower_margin": 0,
    }

    for candidate in candidates:
        if metadata_contains_flower(candidate):
            rejected_reasons["metadata_flower"] += 1
            continue
        red_fraction = float(getattr(candidate, "feijoa_red_fraction", 1.0))
        green_fraction = float(getattr(candidate, "feijoa_green_fraction", 0.0))
        raw_margin = float(getattr(candidate, "feijoa_raw_margin", -1.0))

        # Feijoa flowers have conspicuous red stamens. A small tolerance allows
        # ordinary backgrounds but rejects flower close-ups and mixed flower shots.
        if red_fraction > 0.055:
            rejected_reasons["red_or_pink"] += 1
            continue
        if green_fraction < 0.025:
            rejected_reasons["insufficient_green"] += 1
            continue
        if candidate.clip_positive < 0.205:
            rejected_reasons["clip_fruit_score"] += 1
            continue
        if raw_margin < 0.008 or candidate.clip_margin < 0.012:
            rejected_reasons["fruit_vs_flower_margin"] += 1
            continue
        strict.append(candidate)

    strict.sort(
        key=lambda item: (
            float(getattr(item, "feijoa_raw_margin", -1.0)),
            item.clip_margin,
            item.clip_positive,
        ),
        reverse=True,
    )

    if len(strict) < minimum_needed:
        raise RuntimeError(
            "Fruit-only feijoa filtering retained "
            f"{len(strict)} images; {minimum_needed} are required. "
            f"Rejected counts: {rejected_reasons}. "
            "The builder refuses to add flower images to reach the target."
        )

    pool_size = min(len(strict), max(180, minimum_needed * 2))
    pool = strict[:pool_size]
    v5.v3.REVIEW_POOLS["feijoa"] = pool
    return pool, {
        "policy": "fruit_only",
        "kept": len(pool),
        "strict_eligible": len(strict),
        "rejected_reasons": rejected_reasons,
        "minimum_raw_fruit_margin": min(
            float(getattr(item, "feijoa_raw_margin", 0.0)) for item in pool
        ),
        "maximum_red_fraction": max(
            float(getattr(item, "feijoa_red_fraction", 0.0)) for item in pool
        ),
    }


base.content_filter = content_filter


def assign_splits(candidates: list[base.Candidate], seed: int) -> None:
    random.Random(seed).shuffle(candidates)
    train_count = round(len(candidates) * 0.70)
    for index, candidate in enumerate(candidates):
        candidate.split = "train" if index < train_count else "test"


base.assign_splits = assign_splits


def validate_dataset(
    selected_by_class: dict[str, list[base.Candidate]],
    dataset_root: Path,
    required_count: int,
) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    all_sha: list[str] = []

    expected_train = round(required_count * 0.70)
    expected_test = required_count - expected_train

    for class_name, selected in selected_by_class.items():
        counts[class_name] = {}
        if len(selected) != required_count:
            errors.append(
                f"{class_name}: expected {required_count} images, got {len(selected)}"
            )
        for split, expected in (("train", expected_train), ("test", expected_test)):
            files = sorted((dataset_root / split / class_name).glob("*.jpg"))
            counts[class_name][split] = len(files)
            if len(files) != expected:
                errors.append(
                    f"{class_name}/{split}: expected {expected}, got {len(files)}"
                )
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB":
                            errors.append(
                                f"{path}: image mode is {image.mode}, expected RGB"
                            )
                        if min(image.size) < 180:
                            errors.append(f"{path}: image too small: {image.size}")
                    all_sha.append(hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception as exc:
                    errors.append(f"{path}: unreadable image: {exc}")

        val_files = list((dataset_root / "val" / class_name).glob("*.jpg"))
        if val_files:
            errors.append(f"{class_name}: unexpected validation files: {len(val_files)}")

    if len(all_sha) != len(set(all_sha)):
        errors.append("Exact duplicate files detected in final dataset")

    source_keys: list[str] = []
    for selected in selected_by_class.values():
        source_keys.extend(base.stable_key(item) for item in selected)
    if len(source_keys) != len(set(source_keys)):
        errors.append("A source record is reused more than once")

    return {
        "status": "ok" if not errors else "failed",
        "split": {"train": "70%", "test": "30%"},
        "required_per_class": required_count,
        "total_images": len(all_sha),
        "unique_sha256": len(set(all_sha)),
        "counts": counts,
        "errors": errors,
    }


base.validate_dataset = validate_dataset


def build_readme(count: int, split_counts: dict[str, dict[str, int]]) -> str:
    return f"""# Fruit-only classification dataset: feijoa, carrot, watermelon

The archive contains **{count} independent source photographs per class**.
The `feijoa` class contains photographs with a clearly visible feijoa fruit;
flower-only photographs, leaves without fruit, and whole shrubs without a
visible fruit are rejected.

Classes:

- `feijoa`
- `carrot`
- `watermelon`

Split per class:

- train: {split_counts['feijoa']['train']} images (70%)
- test: {split_counts['feijoa']['test']} images (30%)

Directory layout:

```text
dataset/
├── train/
│   ├── feijoa/
│   ├── carrot/
│   └── watermelon/
├── test/
│   ├── feijoa/
│   ├── carrot/
│   └── watermelon/
├── manifest.csv
├── validation.json
├── duplicate_rejections.json
├── contact_sheets/
└── LICENSE_NOTICE.md
```

No rotated, mirrored, recolored, or cropped derivative is counted as a new
sample. Augmentation should be applied dynamically only to the training split.
The test split must not be augmented.

Quality controls include SHA-256 duplicate detection, perceptual hashes, CLIP
near-duplicate filtering, a dedicated feijoa-fruit-versus-feijoa-flower CLIP
comparison, a red-flower color filter, and visual contact sheets.

Attribution and source information are stored in `manifest.csv` and
`snacks_credits.csv`.
"""


base.build_readme = build_readme


if __name__ == "__main__":
    raise SystemExit(base.main())
