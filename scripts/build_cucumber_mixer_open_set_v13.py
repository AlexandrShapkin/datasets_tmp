#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import requests
from PIL import ImageOps, ImageStat

import build_cucumber_mixer_open_set as original
import build_cucumber_mixer_open_set_v8 as v8
import build_cucumber_mixer_open_set_v11 as v11

builder = v8.builder
base = builder.base


def api_and_openverse_candidates(class_name: str) -> list[base.Candidate]:
    candidates: list[base.Candidate] = []
    categories = original.TARGETS[class_name]["categories"]
    try:
        candidates.extend(original.commons_category_search(class_name, categories))
    except Exception as exc:
        print(f"[api-source] category search failed: {exc}", flush=True)

    client = base.HttpClient()
    for query in original.TARGETS[class_name]["queries"]:
        try:
            candidates.extend(base.commons_search(client, class_name, query, 250))
        except Exception as exc:
            print(f"[api-source] Commons query {query!r} failed: {exc}", flush=True)
        try:
            candidates.extend(original.openverse_search(class_name, query, 220))
        except Exception as exc:
            print(f"[api-source] Openverse query {query!r} failed: {exc}", flush=True)

    result = base.unique_candidates(candidates)
    print(f"[api-source] {class_name}: unique={len(result)}", flush=True)
    return result


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name == "cucumber":
        return v8.kaggle_cucumber_candidates()

    if class_name != "concrete_mixer":
        raise ValueError(f"Unsupported class: {class_name}")

    api_candidates = api_and_openverse_candidates(class_name)
    redirect_candidates = v8.v7.commons_html_candidates(
        class_name,
        v8.v7.MIXER_CATEGORIES,
    )
    thumbnail_candidates = v11.v9.category_candidates(
        class_name,
        v11.v9.MIXER_CATEGORIES,
    )

    # API/Openverse URLs tend to be direct and stable. Process them before the
    # HTML fallbacks. Duplicates are still removed after image decoding.
    combined = base.unique_candidates(
        api_candidates + redirect_candidates + thumbnail_candidates
    )
    print(
        f"[combined-mixer-source] api_openverse={len(api_candidates)}, "
        f"redirects={len(redirect_candidates)}, thumbnails={len(thumbnail_candidates)}, "
        f"combined_urls={len(combined)}",
        flush=True,
    )
    if len(combined) < 300:
        raise RuntimeError(f"Only {len(combined)} mixer candidate URLs were collected")
    return combined


builder.collect_target_candidates = collect_target_candidates


def relaxed_download_one(
    candidate: base.Candidate,
    download_root: Path,
    index: int,
) -> base.Candidate | None:
    if candidate.image_url.startswith("file://"):
        return v8.v7.v6.v5.v4.v3.local_file_download(
            candidate, download_root, index
        )

    headers = {
        "User-Agent": builder.USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": candidate.landing_url or "https://commons.wikimedia.org/",
    }
    for attempt in range(6):
        try:
            response = requests.get(
                candidate.image_url,
                headers=headers,
                timeout=45,
                allow_redirects=True,
            )
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(16, 2 ** attempt))
                continue
            if 400 <= response.status_code < 500:
                return None
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get("content-type", "").lower()
            if len(content) < 2500 or len(content) > 35_000_000:
                return None
            if content_type and "image" not in content_type and "octet-stream" not in content_type:
                return None

            image = base.decode_image(content)
            if min(image.size) < 96:
                return None
            if max(image.size) / max(1, min(image.size)) > 5.0:
                return None
            if base.image_entropy(image) < 2.75:
                return None
            stat = ImageStat.Stat(image.resize((64, 64)))
            if max(stat.var) < 12:
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
        except requests.RequestException:
            if attempt < 5:
                time.sleep(min(12, 2 ** attempt))
                continue
            return None
        except Exception:
            return None
    return None


def multi_source_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    if not candidates:
        return []

    class_name = candidates[0].class_name
    if class_name == "cucumber":
        # Local Kaggle files are fast and deterministic.
        selected = candidates[: min(max_downloads, 320)]
        result: list[base.Candidate] = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    relaxed_download_one, candidate, download_root, index
                ): candidate
                for index, candidate in enumerate(selected)
            }
            for future in as_completed(futures):
                item = future.result()
                if item is not None:
                    result.append(item)
        result.sort(key=base.stable_key)
        print(f"[download] cucumber valid={len(result)}", flush=True)
        return result

    selected = candidates[: min(max_downloads, len(candidates), 2400)]
    result: list[base.Candidate] = []
    target_valid = 145
    batch_size = 48

    # Low concurrency avoids Wikimedia throttling. The ordered source list means
    # stable API/Openverse URLs are attempted before HTML fallbacks.
    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    relaxed_download_one,
                    candidate,
                    download_root,
                    start + offset,
                ): candidate
                for offset, candidate in enumerate(chunk)
            }
            for future in as_completed(futures):
                item = future.result()
                if item is not None:
                    result.append(item)

        print(
            f"[download] concrete_mixer attempted="
            f"{min(start + len(chunk), len(selected))}/{len(selected)}, "
            f"valid={len(result)}",
            flush=True,
        )
        if len(result) >= target_valid:
            break

    result.sort(key=base.stable_key)
    return result


base.download_candidates = multi_source_download_candidates


def main() -> int:
    return v8.v7.v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
