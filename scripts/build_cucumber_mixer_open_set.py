#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
import time
import urllib.parse
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import imagehash
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import CLIPModel, CLIPProcessor

import build_dataset_core as base


SEED = 20260701
MODEL_NAME = "openai/clip-vit-base-patch32"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
SNACKS_ZIP = "https://huggingface.co/datasets/Matthijs/snacks/resolve/main/images.zip"
SNACKS_CREDITS = "https://huggingface.co/datasets/Matthijs/snacks/resolve/main/credits.csv"
SNACKS_PAGE = "https://huggingface.co/datasets/Matthijs/snacks"
USER_AGENT = "AlexandrShapkin-open-set-dataset-builder/1.0 educational project"

TARGETS: dict[str, dict[str, Any]] = {
    "cucumber": {
        "queries": [
            "cucumber vegetable",
            "fresh cucumbers",
            "cucumber fruit",
            "sliced cucumber",
            "Cucumis sativus fruit",
            "cucumbers market",
        ],
        "categories": [
            "Cucumbers",
            "Cucumis sativus",
            "Sliced cucumbers",
            "Cucumbers in markets",
        ],
        "positive": [
            "a clear photograph of a cucumber vegetable",
            "a photograph of fresh green cucumbers",
            "a photograph of sliced cucumber",
            "a cucumber fruit occupying a substantial part of the image",
        ],
        "negative": [
            "a photograph of a zucchini",
            "a photograph of a melon",
            "a photograph of a pickle jar",
            "cucumber leaves or flowers without visible cucumber fruit",
            "a drawing or illustration of a cucumber",
            "a packaged cosmetic product",
        ],
    },
    "concrete_mixer": {
        "queries": [
            "concrete mixer machine",
            "cement mixer machine",
            "concrete mixer truck",
            "cement mixer truck",
            "portable concrete mixer",
            "бетономешалка",
        ],
        "categories": [
            "Concrete mixers",
            "Concrete mixer trucks",
            "Cement mixers",
        ],
        "positive": [
            "a clear photograph of a concrete mixer machine",
            "a photograph of a cement mixer with a rotating drum",
            "a photograph of a concrete mixer truck",
            "a portable concrete mixer occupying a substantial part of the image",
        ],
        "negative": [
            "a photograph of a dump truck without a mixing drum",
            "a photograph of an excavator",
            "a photograph of a crane",
            "a photograph of a garbage truck",
            "bags of cement or a concrete wall without a mixer",
            "a washing machine",
            "a drawing or toy concrete mixer",
        ],
    },
}

OTHER_CLASSES = [
    "apple",
    "banana",
    "cake",
    "cookie",
    "doughnut",
    "grape",
    "hot dog",
    "pineapple",
    "strawberry",
    "waffle",
]


def allowed_license(value: str | None) -> bool:
    text = " ".join((value or "").lower().replace("_", " ").split())
    if not text:
        return False
    tokens = (
        "cc0",
        "pdm",
        "public domain",
        "publicdomain",
        "cc by",
        "cc-by",
        "by-sa",
        "attribution",
        "creativecommons.org/licenses/by/",
        "creativecommons.org/licenses/by-sa/",
        "creativecommons.org/publicdomain/",
    )
    return text in {"by", "by-sa"} or any(token in text for token in tokens)


base.license_is_allowed = allowed_license


def strip_html(value: Any) -> str:
    return base.strip_html(value)


def get_json(url: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=50,
            )
            if response.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last = exc
            time.sleep(min(10, 2 ** attempt))
    raise RuntimeError(f"GET failed: {url}: {last}")


def commons_pageids_to_candidates(
    class_name: str,
    pageids: list[str],
    query: str,
) -> list[base.Candidate]:
    output: list[base.Candidate] = []
    for start in range(0, len(pageids), 50):
        payload = get_json(
            COMMONS_API,
            {
                "action": "query",
                "format": "json",
                "pageids": "|".join(pageids[start : start + 50]),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": 1400,
            },
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/") or mime in {"image/svg+xml", "image/gif"}:
                continue
            metadata = info.get("extmetadata", {})
            license_name = (
                metadata.get("LicenseShortName", {}).get("value")
                or metadata.get("UsageTerms", {}).get("value")
                or ""
            )
            if not allowed_license(license_name):
                continue
            image_url = info.get("thumburl") or info.get("url")
            landing_url = info.get("descriptionurl")
            if not image_url or not landing_url:
                continue
            output.append(
                base.Candidate(
                    class_name=class_name,
                    source="wikimedia_commons",
                    query=query,
                    image_url=image_url,
                    landing_url=landing_url,
                    title=strip_html(
                        metadata.get("ObjectName", {}).get("value")
                        or page.get("title", "")
                    ),
                    creator=strip_html(metadata.get("Artist", {}).get("value", "")),
                    license=strip_html(license_name),
                    license_url=metadata.get("LicenseUrl", {}).get("value", ""),
                    source_id=str(page.get("pageid", "")),
                )
            )
    return output


def commons_category_members(category: str, max_depth: int = 2) -> list[tuple[str, str]]:
    queue: list[tuple[str, int]] = [(category, 0)]
    visited: set[str] = set()
    files: dict[str, str] = {}

    while queue:
        current, depth = queue.pop(0)
        normalized = current.removeprefix("Category:")
        if normalized in visited:
            continue
        visited.add(normalized)
        continuation: str | None = None

        while True:
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{normalized}",
                "cmtype": "file|subcat",
                "cmlimit": 500,
            }
            if continuation:
                params["cmcontinue"] = continuation
            payload = get_json(COMMONS_API, params)
            for member in payload.get("query", {}).get("categorymembers", []):
                namespace = int(member.get("ns", -1))
                if namespace == 6:
                    files[str(member["pageid"])] = normalized
                elif namespace == 14 and depth < max_depth:
                    queue.append((str(member.get("title", "")), depth + 1))
            continuation = payload.get("continue", {}).get("cmcontinue")
            if not continuation:
                break

    return list(files.items())


def commons_category_search(class_name: str, categories: list[str]) -> list[base.Candidate]:
    output: list[base.Candidate] = []
    for category in categories:
        try:
            members = commons_category_members(category)
            print(f"[commons-category] {class_name}/{category}: {len(members)} files")
            pageids = [pageid for pageid, _ in members]
            output.extend(
                commons_pageids_to_candidates(
                    class_name,
                    pageids,
                    query=f"Category:{category}",
                )
            )
        except Exception as exc:
            print(f"[commons-category] failed {category}: {exc}", file=sys.stderr)
    return output


def openverse_search(class_name: str, query: str, limit: int = 300) -> list[base.Candidate]:
    output: list[base.Candidate] = []
    for page in range(1, 7):
        try:
            payload = get_json(
                OPENVERSE_API,
                {
                    "q": query,
                    "page_size": 80,
                    "page": page,
                    "license": "cc0,pdm,by,by-sa",
                    "mature": "false",
                },
            )
        except Exception as exc:
            print(f"[openverse] {query!r}: {exc}", file=sys.stderr)
            break
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            license_name = item.get("license") or ""
            if not allowed_license(license_name):
                continue
            image_url = item.get("thumbnail") or item.get("url")
            landing_url = item.get("foreign_landing_url") or item.get("detail_url")
            if not image_url or not landing_url:
                continue
            output.append(
                base.Candidate(
                    class_name=class_name,
                    source="openverse",
                    query=query,
                    image_url=image_url,
                    landing_url=landing_url,
                    title=item.get("title") or "",
                    creator=item.get("creator") or "",
                    license=license_name,
                    license_url=item.get("license_url") or "",
                    source_id=str(item.get("id") or ""),
                )
            )
            if len(output) >= limit:
                return output
    return output


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    config = TARGETS[class_name]
    candidates: list[base.Candidate] = []
    candidates.extend(commons_category_search(class_name, config["categories"]))

    client = base.HttpClient()
    for query in config["queries"]:
        try:
            candidates.extend(base.commons_search(client, class_name, query, 250))
        except Exception as exc:
            print(f"[commons-search] {query!r}: {exc}", file=sys.stderr)
        candidates.extend(openverse_search(class_name, query, 250))

    unique = base.unique_candidates(candidates)
    random.Random(SEED + len(class_name)).shuffle(unique)
    print(f"[collect] {class_name}: raw={len(candidates)}, unique_urls={len(unique)}")
    return unique


def clip_score(
    candidates: list[base.Candidate],
    class_name: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 24,
) -> list[base.Candidate]:
    config = TARGETS[class_name]
    positive = base.clip_text_features(model, processor, config["positive"], device)
    negative = base.clip_text_features(model, processor, config["negative"], device)
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
        positive_scores = (embeddings @ positive.T).max(axis=1)
        negative_scores = (embeddings @ negative.T).max(axis=1)

        for candidate, embedding, pos, neg in zip(
            valid, embeddings, positive_scores, negative_scores
        ):
            candidate.clip_positive = float(pos)
            candidate.clip_negative = float(neg)
            candidate.clip_margin = float(pos - neg)
            candidate.clip_embedding = embedding.tolist()
            output.append(candidate)
        print(f"[clip] {class_name}: {len(output)}/{len(candidates)}")
    return output


def select_target(candidates: list[base.Candidate], count: int) -> list[base.Candidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.clip_margin, item.clip_positive),
        reverse=True,
    )
    pool = [
        item
        for item in ordered
        if item.clip_positive >= 0.19 and item.clip_margin >= -0.025
    ]
    if len(pool) < max(count, 120):
        pool = ordered[: max(count * 3, 180)]
    unique, rejected = base.embedding_deduplicate(pool, cosine_threshold=0.993)
    if len(unique) < count:
        unique, rejected = base.embedding_deduplicate(pool, cosine_threshold=0.997)
    if len(unique) < count:
        raise RuntimeError(f"Only {len(unique)} usable candidates remain; need {count}")
    return base.diverse_select(unique, count)


def ensure_snacks(work: Path) -> tuple[Path, Path]:
    root = work / "snacks"
    archive_path = root / "images.zip"
    credits_path = root / "credits.csv"
    marker = root / ".extracted"
    root.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists() or archive_path.stat().st_size < 50_000_000:
        print("[snacks] downloading images.zip")
        with requests.get(SNACKS_ZIP, stream=True, timeout=180) as response:
            response.raise_for_status()
            temporary = archive_path.with_suffix(".part")
            with temporary.open("wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        file.write(chunk)
            temporary.replace(archive_path)

    if not marker.exists():
        print("[snacks] extracting")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
        marker.write_text("ok\n", encoding="utf-8")

    if not credits_path.exists():
        response = requests.get(SNACKS_CREDITS, timeout=90)
        response.raise_for_status()
        credits_path.write_bytes(response.content)

    return root, credits_path


def choose_other_images(work: Path, per_class: int = 4) -> list[tuple[str, Path]]:
    root, _ = ensure_snacks(work)
    chosen: list[tuple[str, Path]] = []
    for class_index, class_name in enumerate(OTHER_CLASSES):
        files: list[Path] = []
        for split in ("train", "val", "test"):
            files.extend(root.glob(f"**/{split}/{class_name}/*.jpg"))
            files.extend(root.glob(f"**/{split}/{class_name}/*.jpeg"))
        files = sorted(set(path.resolve() for path in files))
        if len(files) < per_class:
            raise RuntimeError(f"Other class {class_name!r} has only {len(files)} images")
        random.Random(SEED + class_index * 101).shuffle(files)
        chosen.extend((class_name, path) for path in files[:per_class])
    return chosen


def save_clean_jpeg(source: Path, destination: Path) -> tuple[int, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        if min(image.size) < 224:
            scale = 224 / min(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        image = base.prepare_image(image, max_side=1280)
        image.save(destination, "JPEG", quality=94, optimize=True, subsampling=0, exif=b"")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return image.width, image.height, digest


def contact_sheet(paths: list[tuple[str, Path]], destination: Path, title: str) -> None:
    columns = 10
    cell = 170
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * cell, 48 + rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill="black", font=font)
    for index, (label, path) in enumerate(paths):
        row, column = divmod(index, columns)
        x0 = column * cell
        y0 = 48 + row * cell
        with Image.open(path) as image:
            image.load()
            thumb = ImageOps.contain(
                ImageOps.exif_transpose(image).convert("RGB"),
                (cell - 8, cell - 28),
                Image.Resampling.LANCZOS,
            )
        x = x0 + (cell - thumb.width) // 2
        y = y0 + 2
        sheet.paste(thumb, (x, y))
        draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), outline="gray")
        draw.text((x0 + 4, y0 + cell - 20), f"{index + 1:03d} {label}", fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=90, optimize=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    work = Path(".work_open_set")
    dist = Path("dist_open_set")
    if work.exists():
        shutil.rmtree(work)
    if dist.exists():
        shutil.rmtree(dist)
    work.mkdir(parents=True)
    dist.mkdir(parents=True)

    downloaded_by_class: dict[str, list[base.Candidate]] = {}
    for class_name in TARGETS:
        collected = collect_target_candidates(class_name)
        downloaded = base.download_candidates(
            collected,
            work / "downloads",
            max_downloads=min(1000, len(collected)),
            workers=20,
        )
        unique, rejected = base.preclip_deduplicate(downloaded)
        print(
            f"[dedup] {class_name}: downloaded={len(downloaded)}, "
            f"unique={len(unique)}, rejected={len(rejected)}"
        )
        if len(unique) < 80:
            raise RuntimeError(f"{class_name}: only {len(unique)} unique downloads; need 80")
        downloaded_by_class[class_name] = unique

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).eval().to(device)

    selected: dict[str, list[base.Candidate]] = {}
    for class_name in TARGETS:
        scored = clip_score(
            downloaded_by_class[class_name],
            class_name,
            model,
            processor,
            device,
        )
        selected[class_name] = select_target(scored, 80)
        print(f"[select] {class_name}: {len(selected[class_name])}")

    dataset_root = dist / "dataset"
    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    all_hashes: set[str] = set()

    target_test_items: list[tuple[str, Path, str, str]] = []
    for class_index, class_name in enumerate(TARGETS):
        items = selected[class_name][:]
        random.Random(SEED + class_index * 1009).shuffle(items)
        train_items = items[:40]
        val_items = items[40:50]
        test_items = items[50:80]

        for split, split_items in (("train", train_items), ("val", val_items)):
            for index, candidate in enumerate(split_items, start=1):
                filename = f"{class_name}_{index:03d}.jpg"
                destination = dataset_root / split / class_name / filename
                width, height, digest = save_clean_jpeg(Path(candidate.local_path), destination)
                if digest in all_hashes:
                    raise RuntimeError(f"Duplicate final image: {candidate.local_path}")
                all_hashes.add(digest)
                train_records.append(
                    {
                        "relative_path": destination.relative_to(dataset_root).as_posix(),
                        "class": class_name,
                        "split": split,
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

    other_items = choose_other_images(work, per_class=4)
    private_test_items: list[tuple[str, str, Path, str, str]] = []
    for class_name, path, source, landing in target_test_items:
        private_test_items.append((class_name, class_name, path, source, landing))
    for source_class, path in other_items:
        private_test_items.append(("other", source_class, path, "matthijs_snacks", SNACKS_PAGE))

    random.Random(SEED + 99991).shuffle(private_test_items)
    ground_truth_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for index, (label, source_class, source_path, source, landing) in enumerate(
        private_test_items,
        start=1,
    ):
        filename = f"image_{index:04d}.jpg"
        destination = dataset_root / "test" / filename
        width, height, digest = save_clean_jpeg(source_path, destination)
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

This dataset is intended for two models: a conventional CNN and Ultralytics
YOLO Classification.

## Training design

The networks are trained only on two known classes:

- `cucumber`
- `concrete_mixer`

The 100 known training images are split into 80 training images and 20
validation images:

- train: 40 cucumber + 40 concrete mixer
- val: 10 cucumber + 10 concrete mixer

## Test design

The test directory contains 100 unlabeled files in one flat directory:

- 30 cucumber
- 30 concrete mixer
- 40 images from unrelated categories (`other`)

The `other` images are not used in training. Because the models have only two
trained outputs, `other` must be implemented as rejection by confidence:

```python
if max_probability < threshold:
    predicted_class = "other"
else:
    predicted_class = known_classes[argmax_probability]
```

Do not put `ground_truth_test.csv` inside the dataset or pass it to the model.
Use it only after inference to calculate metrics.
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8")

    validation = {
        "status": "ok",
        "known_training_dataset_total": 100,
        "train": {"cucumber": 40, "concrete_mixer": 40},
        "val": {"cucumber": 10, "concrete_mixer": 10},
        "unlabeled_test_total": 100,
        "private_test_distribution": {
            "cucumber": 30,
            "concrete_mixer": 30,
            "other": 40,
        },
        "other_used_for_training": False,
        "unique_final_sha256": len(all_hashes),
        "expected_unique_final_sha256": 200,
        "test_labels_exposed_in_dataset": False,
    }
    (dataset_root / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(
        dist / "ground_truth_test.csv",
        ground_truth_rows,
        ["filename", "class"],
    )
    write_csv(
        dist / "source_manifest_private.csv",
        train_records + source_rows,
        sorted(set().union(*(row.keys() for row in train_records + source_rows))),
    )

    contact_sheet(
        [("cucumber", Path(item.local_path)) for item in selected["cucumber"]],
        dist / "contact_sheets" / "cucumber_selected.jpg",
        "Cucumber: 80 selected unique photographs",
    )
    contact_sheet(
        [("concrete_mixer", Path(item.local_path)) for item in selected["concrete_mixer"]],
        dist / "contact_sheets" / "concrete_mixer_selected.jpg",
        "Concrete mixer: 80 selected unique photographs",
    )
    contact_sheet(
        [(label, path) for label, _, path, _, _ in private_test_items],
        dist / "contact_sheets" / "test_private.jpg",
        "Private test review: labels shown only for manual validation",
    )

    archive_path = dist / "cucumber_concrete_mixer_open_set.zip"
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
        test_names = [name for name in names if name.startswith("dataset/test/") and name.endswith(".jpg")]
        if len(test_names) != 100:
            raise RuntimeError(f"Expected 100 test images, got {len(test_names)}")
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
