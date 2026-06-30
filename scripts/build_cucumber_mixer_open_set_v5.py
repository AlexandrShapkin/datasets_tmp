#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import requests
from PIL import ImageOps, ImageStat

import build_cucumber_mixer_open_set_v4 as v4

builder = v4.builder
base = builder.base


def efficient_download_one(candidate, download_root: Path, index: int):
    if candidate.image_url.startswith("file://"):
        return v4.v3.local_file_download(candidate, download_root, index)

    headers = {
        "User-Agent": builder.USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": candidate.landing_url or "https://commons.wikimedia.org/",
    }
    for attempt in range(4):
        try:
            response = requests.get(candidate.image_url, headers=headers, timeout=35, allow_redirects=True)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(8, 1.5 * (2 ** attempt)))
                continue
            if 400 <= response.status_code < 500:
                return None
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            content = response.content
            if len(content) < 4000 or len(content) > 30000000:
                return None
            if content_type and "image" not in content_type and "octet-stream" not in content_type:
                return None
            image = base.decode_image(content)
            if min(image.size) < 150 or max(image.size) / max(1, min(image.size)) > 4.5:
                return None
            if base.image_entropy(image) < 3.1:
                return None
            stat = ImageStat.Stat(image.resize((64, 64)))
            if max(stat.var) < 18:
                return None
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
            candidate.mirror_phash = str(imagehash.phash(ImageOps.mirror(image), hash_size=16))
            return candidate
        except requests.RequestException:
            if attempt < 3:
                time.sleep(min(6, 2 ** attempt))
                continue
            return None
        except Exception:
            return None
    return None


def efficient_adaptive_download(candidates, download_root: Path, max_downloads: int, workers: int):
    if not candidates:
        return []
    class_name = candidates[0].class_name
    target_valid = 118 if class_name == "cucumber" else 135
    maximum = min(max_downloads, 360 if class_name == "cucumber" else 460)
    selected = candidates[:maximum]
    result = []
    batch_size = 64
    for start in range(0, len(selected), batch_size):
        chunk = selected[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
            futures = {
                executor.submit(efficient_download_one, candidate, download_root, start + offset): candidate
                for offset, candidate in enumerate(chunk)
            }
            for future in as_completed(futures):
                item = future.result()
                if item is not None:
                    result.append(item)
        print(f"[efficient-download] {class_name}: attempted={min(start + len(chunk), len(selected))}/{len(selected)}, valid={len(result)}", flush=True)
        if len(result) >= target_valid:
            break
    result.sort(key=base.stable_key)
    return result


base.download_candidates = efficient_adaptive_download

if __name__ == "__main__":
    raise SystemExit(builder.main())
