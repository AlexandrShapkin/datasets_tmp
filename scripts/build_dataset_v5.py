#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
import threading
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps, ImageStat

import build_dataset_v3 as v3

base = v3.base

SNACKS_DATASET_URL = "https://huggingface.co/datasets/Matthijs/snacks/resolve/main/images.zip"
SNACKS_CREDITS_URL = "https://huggingface.co/datasets/Matthijs/snacks/resolve/main/credits.csv"
SNACKS_LANDING_URL = "https://huggingface.co/datasets/Matthijs/snacks"
SNACKS_LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"
SNACKS_CLASSES = {"carrot", "watermelon"}
SNACKS_CACHE = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "matthijs_snacks_source"
SNACKS_LOCK = threading.Lock()
SNACKS_CREDITS_PATH: Path | None = None

ORIGINAL_COLLECT_CANDIDATES = base.collect_candidates
ORIGINAL_DOWNLOAD_ONE = base.download_one
ORIGINAL_WRITE_MANIFEST = base.write_manifest
ORIGINAL_BUILD_README = base.build_readme


def ensure_snacks_source(client: base.HttpClient) -> tuple[Path, Path]:
    global SNACKS_CREDITS_PATH
    with SNACKS_LOCK:
        extracted_marker = SNACKS_CACHE / ".extracted"
        archive_path = SNACKS_CACHE / "images.zip"
        credits_path = SNACKS_CACHE / "credits.csv"
        SNACKS_CACHE.mkdir(parents=True, exist_ok=True)

        if not extracted_marker.exists():
            if not archive_path.exists() or archive_path.stat().st_size < 50_000_000:
                print(f"[snacks] downloading {SNACKS_DATASET_URL}")
                response = client.get(SNACKS_DATASET_URL, timeout=120, stream=True, retries=6)
                temporary = archive_path.with_suffix(".part")
                with temporary.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
                temporary.replace(archive_path)
                print(f"[snacks] downloaded {archive_path.stat().st_size} bytes")

            print("[snacks] extracting images.zip")
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(SNACKS_CACHE)
            extracted_marker.write_text("ok\n", encoding="utf-8")

        if not credits_path.exists() or credits_path.stat().st_size < 1000:
            response = client.get(SNACKS_CREDITS_URL, timeout=90, stream=True, retries=6)
            credits_path.write_bytes(response.content)

        SNACKS_CREDITS_PATH = credits_path
        return SNACKS_CACHE, credits_path


def load_credit_index(path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for raw_row in reader:
                row = {str(key): str(value or "") for key, value in raw_row.items()}
                for value in row.values():
                    normalized = value.replace("\\", "/")
                    basename = normalized.rsplit("/", 1)[-1]
                    stem = Path(basename).stem
                    if basename.lower().endswith((".jpg", ".jpeg", ".png")):
                        index.setdefault(basename, row)
                        index.setdefault(stem, row)
    except Exception as exc:
        print(f"[snacks] could not index credits.csv: {exc}")
    return index


def first_matching(row: dict[str, str], fragments: tuple[str, ...]) -> str:
    for key, value in row.items():
        lowered = key.lower()
        if any(fragment in lowered for fragment in fragments) and value:
            return value
    return ""


def snack_candidates(
    client: base.HttpClient,
    class_name: str,
) -> list[base.Candidate]:
    root, credits_path = ensure_snacks_source(client)
    credit_index = load_credit_index(credits_path)
    paths: list[Path] = []
    for split in ("train", "val", "test"):
        paths.extend(root.glob(f"**/{split}/{class_name}/*.jpg"))
        paths.extend(root.glob(f"**/{split}/{class_name}/*.jpeg"))
    paths = sorted(set(path.resolve() for path in paths))
    if len(paths) < 100:
        raise RuntimeError(
            f"Matthijs/snacks extraction contains only {len(paths)} {class_name} images"
        )

    random.Random(2026062900 + sum(ord(char) for char in class_name)).shuffle(paths)
    candidates: list[base.Candidate] = []
    for path in paths:
        row = credit_index.get(path.name, credit_index.get(path.stem, {}))
        creator = first_matching(row, ("author", "creator", "attribution", "photographer"))
        original_url = first_matching(row, ("original", "url", "source"))
        relative = path.relative_to(root).as_posix()
        candidates.append(
            base.Candidate(
                class_name=class_name,
                source="matthijs_snacks_openimages",
                query=class_name,
                image_url=path.as_uri(),
                landing_url=original_url or SNACKS_LANDING_URL,
                title=f"{class_name} — {path.name}",
                creator=creator or "See snacks_credits.csv",
                license="CC BY 2.0",
                license_url=SNACKS_LICENSE_URL,
                source_id=relative,
            )
        )
    print(f"[snacks] {class_name}: {len(candidates)} local source photographs")
    return candidates


def collect_candidates(
    client: base.HttpClient,
    class_name: str,
    source_limit: int,
) -> list[base.Candidate]:
    if class_name in SNACKS_CLASSES:
        return snack_candidates(client, class_name)
    return ORIGINAL_COLLECT_CANDIDATES(client, class_name, source_limit)


base.collect_candidates = collect_candidates


def local_download_one(
    candidate: base.Candidate,
    download_root: Path,
    index: int,
) -> base.Candidate | None:
    if not candidate.image_url.startswith("file://"):
        return ORIGINAL_DOWNLOAD_ONE(candidate, download_root, index)

    try:
        parsed = urllib.parse.urlparse(candidate.image_url)
        source_path = Path(urllib.parse.unquote(parsed.path))
        content = source_path.read_bytes()
        if len(content) < 8_000 or len(content) > 30_000_000:
            return None
        image = base.decode_image(content)
        width, height = image.size
        if min(width, height) < 180:
            return None
        if max(width, height) / max(1, min(width, height)) > 4.5:
            return None
        if base.image_entropy(image) < 3.4:
            return None
        stat = ImageStat.Stat(image.resize((64, 64)))
        if max(stat.var) < 25:
            return None

        image = base.prepare_image(image)
        class_dir = download_root / candidate.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_dir / f"candidate_{index:05d}.jpg"
        image.save(output_path, "JPEG", quality=94, optimize=True, subsampling=0)

        candidate.local_path = str(output_path)
        candidate.width, candidate.height = image.size
        candidate.sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        candidate.phash = str(imagehash.phash(image, hash_size=16))
        candidate.dhash = str(imagehash.dhash(image, hash_size=16))
        candidate.whash = str(imagehash.whash(image, hash_size=16))
        candidate.mirror_phash = str(
            imagehash.phash(ImageOps.mirror(image), hash_size=16)
        )
        return candidate
    except Exception as exc:
        print(f"[snacks] failed to process {candidate.source_id}: {exc}")
        return None


base.download_one = local_download_one


def write_manifest(candidates: list[base.Candidate], path: Path) -> None:
    ORIGINAL_WRITE_MANIFEST(candidates, path)
    if SNACKS_CREDITS_PATH and SNACKS_CREDITS_PATH.exists():
        shutil.copy2(SNACKS_CREDITS_PATH, path.parent / "snacks_credits.csv")


base.write_manifest = write_manifest


def build_readme(count: int, split_counts: dict[str, dict[str, int]]) -> str:
    text = ORIGINAL_BUILD_README(count, split_counts)
    text += """

## Additional curated source for carrot and watermelon

The carrot and watermelon candidate pools are taken from the
`Matthijs/snacks` dataset, which contains independently sourced Google Open
Images photographs. The source dataset provides 349 carrot photographs and
350 watermelon photographs. Images are licensed CC BY 2.0 and annotations are
licensed CC BY 4.0. Per-image attribution data is preserved in
`snacks_credits.csv`.
"""
    return text


base.build_readme = build_readme


if __name__ == "__main__":
    raise SystemExit(base.main())
