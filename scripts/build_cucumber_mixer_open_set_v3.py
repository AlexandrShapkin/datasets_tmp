#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import kagglehub
from PIL import ImageOps, ImageStat

import build_cucumber_mixer_open_set_v2 as v2

builder = v2.builder
base = builder.base

CUCUMBER_HANDLE = "kritikseth/fruit-and-vegetable-image-recognition"
CUCUMBER_PAGE = "https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition"
CUCUMBER_CACHE = Path(".work_open_set/kaggle_cucumber")

ORIGINAL_COLLECT = builder.collect_target_candidates


def download_kaggle_path(split: str, index: int) -> tuple[str, Path] | None:
    remote = f"{split}/cucumber/Image_{index}.jpg"
    try:
        resolved = kagglehub.dataset_download(
            CUCUMBER_HANDLE,
            path=remote,
            output_dir=str(CUCUMBER_CACHE),
        )
        path = Path(resolved)
        if path.is_file() and path.stat().st_size > 1000:
            return remote, path
        candidate = CUCUMBER_CACHE / remote
        if candidate.exists() and candidate.stat().st_size > 1000:
            return remote, candidate
    except BaseException:
        return None
    return None


def kaggle_cucumber_candidates() -> list[base.Candidate]:
    requests_to_make: list[tuple[str, int]] = []
    requests_to_make.extend(("train", index) for index in range(1, 161))
    requests_to_make.extend(("validation", index) for index in range(1, 41))
    requests_to_make.extend(("test", index) for index in range(1, 41))

    found: list[tuple[str, Path]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_kaggle_path, split, index): (split, index)
            for split, index in requests_to_make
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                found.append(result)
            if completed % 40 == 0 or completed == len(futures):
                print(
                    f"[kaggle-cucumber] completed={completed}/{len(futures)} "
                    f"found={len(found)}",
                    flush=True,
                )

    unique_paths = sorted({path.resolve() for _, path in found})
    if len(unique_paths) < 90:
        raise RuntimeError(
            f"Kaggle cucumber source yielded only {len(unique_paths)} files; need at least 90"
        )
    random.Random(builder.SEED + 77).shuffle(unique_paths)

    candidates: list[base.Candidate] = []
    for path in unique_paths:
        try:
            relative = path.relative_to(CUCUMBER_CACHE.resolve()).as_posix()
        except ValueError:
            relative = path.name
        candidates.append(
            base.Candidate(
                class_name="cucumber",
                source="kaggle_fruit_vegetable_recognition",
                query="cucumber",
                image_url=path.as_uri(),
                landing_url=CUCUMBER_PAGE,
                title=f"Cucumber — {path.name}",
                creator="See Kaggle dataset source",
                license="See Kaggle dataset terms",
                license_url=CUCUMBER_PAGE,
                source_id=relative,
            )
        )
    print(f"[kaggle-cucumber] source candidates={len(candidates)}")
    return candidates


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name == "cucumber":
        return kaggle_cucumber_candidates()
    return ORIGINAL_COLLECT(class_name)


builder.collect_target_candidates = collect_target_candidates


def local_file_download(
    candidate: base.Candidate,
    download_root: Path,
    index: int,
) -> base.Candidate | None:
    if not candidate.image_url.startswith("file://"):
        return v2.robust_download_one(candidate, download_root, index)
    try:
        parsed = urllib.parse.urlparse(candidate.image_url)
        source_path = Path(urllib.parse.unquote(parsed.path))
        content = source_path.read_bytes()
        if len(content) < 3_000 or len(content) > 30_000_000:
            return None
        image = base.decode_image(content)
        if min(image.size) < 120:
            return None
        if max(image.size) / max(1, min(image.size)) > 4.5:
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
                base.Image.Resampling.LANCZOS,
            )
        image = base.prepare_image(image)
        class_dir = download_root / candidate.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / f"candidate_{index:05d}.jpg"
        image.save(destination, "JPEG", quality=94, optimize=True, subsampling=0)

        candidate.local_path = str(destination)
        candidate.width, candidate.height = image.size
        candidate.sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        candidate.phash = str(imagehash.phash(image, hash_size=16))
        candidate.dhash = str(imagehash.dhash(image, hash_size=16))
        candidate.whash = str(imagehash.whash(image, hash_size=16))
        candidate.mirror_phash = str(
            imagehash.phash(ImageOps.mirror(image), hash_size=16)
        )
        return candidate
    except Exception as exc:
        print(f"[local-cucumber] failed {candidate.source_id}: {exc}")
        return None


def mixed_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    selected = candidates[: min(max_downloads, 650)]
    result: list[base.Candidate] = []
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = {
            executor.submit(local_file_download, candidate, download_root, index): candidate
            for index, candidate in enumerate(selected)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            item = future.result()
            if item is not None:
                result.append(item)
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"[mixed-download] completed={completed}/{len(futures)} "
                    f"valid={len(result)}",
                    flush=True,
                )
    result.sort(key=base.stable_key)
    return result


base.download_candidates = mixed_download_candidates


if __name__ == "__main__":
    raise SystemExit(builder.main())
