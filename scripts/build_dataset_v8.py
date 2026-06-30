#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imagehash
import kagglehub
from PIL import Image, ImageOps, ImageStat

import build_dataset_v7 as v7

base = v7.base

FRUITS262_HANDLE = "aelchimminut/fruits262"
FRUITS262_LANDING = "https://www.kaggle.com/datasets/aelchimminut/fruits262"
FRUITS262_CACHE = Path(".work/fruits262_kaggle")
FRUITS262_TARGET_CANDIDATES = 500
FRUITS262_MAX_INDEX = 1800
FRUITS262_BATCH_SIZE = 100
FRUITS262_WORKERS = 16

ORIGINAL_COLLECT_CANDIDATES = base.collect_candidates
ORIGINAL_DOWNLOAD_ONE = base.download_one
ORIGINAL_CONTENT_FILTER = base.content_filter

_download_lock = threading.Lock()


def download_fruits262_file(index: int) -> Path | None:
    remote_path = f"Fruit-262/feijoa/{index}.jpg"
    expected = FRUITS262_CACHE / remote_path
    if expected.exists() and expected.stat().st_size > 1000:
        return expected
    try:
        resolved = kagglehub.dataset_download(
            FRUITS262_HANDLE,
            path=remote_path,
            output_dir=str(FRUITS262_CACHE),
        )
        path = Path(resolved)
        if path.is_file():
            return path
        candidate = path / remote_path
        if candidate.exists():
            return candidate
        if expected.exists():
            return expected
    except BaseException as exc:
        message = str(exc).lower()
        if "not found" not in message and "404" not in message:
            with _download_lock:
                print(f"[fruits262] index={index}: {type(exc).__name__}: {exc}")
    return None


def ensure_fruits262_feijoa() -> list[Path]:
    class_dir = FRUITS262_CACHE / "Fruit-262" / "feijoa"
    class_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(path.stem): path
        for path in class_dir.glob("*.jpg")
        if path.stem.isdigit() and path.stat().st_size > 1000
    }
    print(f"[fruits262] cached feijoa files: {len(existing)}")

    low_yield_batches = 0
    for start in range(0, FRUITS262_MAX_INDEX, FRUITS262_BATCH_SIZE):
        if len(existing) >= FRUITS262_TARGET_CANDIDATES:
            break
        indices = [
            index
            for index in range(start, min(start + FRUITS262_BATCH_SIZE, FRUITS262_MAX_INDEX))
            if index not in existing
        ]
        if not indices:
            continue
        batch_success = 0
        with ThreadPoolExecutor(max_workers=FRUITS262_WORKERS) as executor:
            futures = {executor.submit(download_fruits262_file, index): index for index in indices}
            for future in as_completed(futures):
                index = futures[future]
                path = future.result()
                if path is not None and path.exists():
                    existing[index] = path
                    batch_success += 1
        print(
            f"[fruits262] indices {start}-{start + len(indices) - 1}: "
            f"downloaded={batch_success}, total={len(existing)}"
        )
        if batch_success < 3 and start >= 200:
            low_yield_batches += 1
        else:
            low_yield_batches = 0
        if low_yield_batches >= 3:
            break

    paths = [existing[index] for index in sorted(existing)]
    if len(paths) < 180:
        raise RuntimeError(
            f"Fruits-262 provided only {len(paths)} downloadable feijoa files; "
            "at least 180 are required for strict filtering and deduplication"
        )
    random.Random(20260630).shuffle(paths)
    print(f"[fruits262] usable source files before image validation: {len(paths)}")
    return paths


def fruits262_candidates() -> list[base.Candidate]:
    candidates: list[base.Candidate] = []
    for path in ensure_fruits262_feijoa():
        relative = path.relative_to(FRUITS262_CACHE).as_posix()
        candidates.append(
            base.Candidate(
                class_name="feijoa",
                source="fruits262_kaggle",
                query="feijoa fruit",
                image_url=path.resolve().as_uri(),
                landing_url=FRUITS262_LANDING,
                title=f"Fruits-262 feijoa fruit {path.name}",
                creator="Fruits-262 dataset; see source page",
                license="See Kaggle dataset terms and source metadata",
                license_url=FRUITS262_LANDING,
                source_id=relative,
            )
        )
    return candidates


def collect_candidates(
    client: base.HttpClient,
    class_name: str,
    source_limit: int,
) -> list[base.Candidate]:
    if class_name == "feijoa":
        return fruits262_candidates()
    return ORIGINAL_COLLECT_CANDIDATES(client, class_name, source_limit)


base.collect_candidates = collect_candidates


def fruits262_download_one(
    candidate: base.Candidate,
    download_root: Path,
    index: int,
) -> base.Candidate | None:
    if candidate.source != "fruits262_kaggle":
        return ORIGINAL_DOWNLOAD_ONE(candidate, download_root, index)
    try:
        parsed = urllib.parse.urlparse(candidate.image_url)
        source_path = Path(urllib.parse.unquote(parsed.path))
        content = source_path.read_bytes()
        if len(content) < 2500 or len(content) > 30_000_000:
            return None
        image = base.decode_image(content)
        width, height = image.size
        if min(width, height) < 96:
            return None
        if max(width, height) / max(1, min(width, height)) > 4.5:
            return None
        if base.image_entropy(image) < 3.0:
            return None
        stat = ImageStat.Stat(image.resize((64, 64)))
        if max(stat.var) < 18:
            return None

        if min(image.size) < 224:
            scale = 224 / min(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
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
        candidate.mirror_phash = str(imagehash.phash(ImageOps.mirror(image), hash_size=16))
        return candidate
    except Exception as exc:
        print(f"[fruits262] processing failed {candidate.source_id}: {exc}")
        return None


base.download_one = fruits262_download_one


def content_filter(
    candidates: list[base.Candidate],
    minimum_needed: int,
) -> tuple[list[base.Candidate], dict[str, Any]]:
    if not candidates or candidates[0].class_name != "feijoa":
        return ORIGINAL_CONTENT_FILTER(candidates, minimum_needed)

    eligible: list[base.Candidate] = []
    rejected = {
        "metadata_flower": 0,
        "prominent_red_flower": 0,
        "very_low_fruit_score": 0,
        "stronger_flower_or_nonfruit_score": 0,
    }
    for candidate in candidates:
        if v7.v6.metadata_contains_flower(candidate):
            rejected["metadata_flower"] += 1
            continue
        red_fraction = float(getattr(candidate, "feijoa_red_fraction", 0.0))
        raw_margin = float(getattr(candidate, "feijoa_raw_margin", -1.0))
        if red_fraction > 0.10:
            rejected["prominent_red_flower"] += 1
            continue
        if candidate.clip_positive < 0.185:
            rejected["very_low_fruit_score"] += 1
            continue
        if raw_margin < -0.035:
            rejected["stronger_flower_or_nonfruit_score"] += 1
            continue
        eligible.append(candidate)

    eligible.sort(
        key=lambda item: (
            float(getattr(item, "feijoa_raw_margin", -1.0)),
            item.clip_positive,
            -float(getattr(item, "feijoa_red_fraction", 0.0)),
        ),
        reverse=True,
    )
    if len(eligible) < minimum_needed:
        raise RuntimeError(
            f"Fruits-262 fruit-only filtering retained {len(eligible)} feijoa images; "
            f"{minimum_needed} are required. Rejected: {rejected}"
        )

    pool_size = min(len(eligible), max(220, minimum_needed * 2))
    pool = eligible[:pool_size]
    v7.v6.v5.v3.REVIEW_POOLS["feijoa"] = pool
    return pool, {
        "policy": "Fruits-262 manual-fruit-source plus flower rejection",
        "eligible": len(eligible),
        "kept": len(pool),
        "rejected": rejected,
        "source": FRUITS262_LANDING,
    }


base.content_filter = content_filter


if __name__ == "__main__":
    raise SystemExit(base.main())
