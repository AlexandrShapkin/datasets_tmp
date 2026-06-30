#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import requests
from PIL import ImageOps, ImageStat

import build_cucumber_mixer_open_set as builder

base = builder.base


def robust_download_one(
    candidate: base.Candidate,
    download_root: Path,
    index: int,
) -> base.Candidate | None:
    headers = {
        "User-Agent": builder.USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": candidate.landing_url or "https://commons.wikimedia.org/",
    }
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(
                candidate.image_url,
                headers=headers,
                timeout=50,
                allow_redirects=True,
            )
            if response.status_code in {429, 502, 503, 504}:
                time.sleep(min(20, 2 ** attempt * 1.5))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            content = response.content
            if len(content) < 5_000 or len(content) > 30_000_000:
                return None
            if content_type and "image" not in content_type and "octet-stream" not in content_type:
                return None

            image = base.decode_image(content)
            width, height = image.size
            if min(width, height) < 160:
                return None
            if max(width, height) / max(1, min(width, height)) > 4.5:
                return None
            if base.image_entropy(image) < 3.2:
                return None
            stat = ImageStat.Stat(image.resize((64, 64)))
            if max(stat.var) < 20:
                return None

            image = base.prepare_image(image)
            class_dir = download_root / candidate.class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            path = class_dir / f"candidate_{index:05d}.jpg"
            image.save(path, "JPEG", quality=94, optimize=True, subsampling=0)

            candidate.local_path = str(path)
            candidate.width, candidate.height = image.size
            candidate.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            candidate.phash = str(imagehash.phash(image, hash_size=16))
            candidate.dhash = str(imagehash.dhash(image, hash_size=16))
            candidate.whash = str(imagehash.whash(image, hash_size=16))
            candidate.mirror_phash = str(
                imagehash.phash(ImageOps.mirror(image), hash_size=16)
            )
            return candidate
        except Exception as exc:
            last_error = exc
            time.sleep(min(15, 2 ** attempt))
    return None


def robust_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    selected = candidates[: min(max_downloads, 650)]
    result: list[base.Candidate] = []
    actual_workers = min(workers, 8)
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {
            executor.submit(robust_download_one, candidate, download_root, index): candidate
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
                    f"[robust-download] completed={completed}/{len(futures)} "
                    f"valid={len(result)}",
                    flush=True,
                )
    result.sort(key=base.stable_key)
    return result


base.download_one = robust_download_one
base.download_candidates = robust_download_candidates


if __name__ == "__main__":
    raise SystemExit(builder.main())
