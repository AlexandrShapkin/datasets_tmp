#!/usr/bin/env python3
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import build_cucumber_mixer_open_set_v8 as v8
import build_cucumber_mixer_open_set_v11 as v11

builder = v8.builder
base = builder.base


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name == "cucumber":
        # The dedicated Vegetable Image Dataset contains 1,400 cucumber files and
        # previously yielded 256 valid unique downloads in CI.
        return v8.kaggle_cucumber_candidates()

    if class_name == "concrete_mixer":
        # Combine two independent Wikimedia access paths. Some files work only
        # through Special:Redirect while others work only through an existing
        # thumbnail URL published in the category page's srcset.
        redirect_candidates = v8.v7.commons_html_candidates(
            class_name,
            v8.v7.MIXER_CATEGORIES,
        )
        thumbnail_candidates = v11.v9.category_candidates(
            class_name,
            v11.v9.MIXER_CATEGORIES,
        )
        combined = base.unique_candidates(redirect_candidates + thumbnail_candidates)
        random.Random(builder.SEED + 12012).shuffle(combined)
        print(
            f"[combined-mixer-source] redirects={len(redirect_candidates)}, "
            f"thumbnails={len(thumbnail_candidates)}, combined={len(combined)}",
            flush=True,
        )
        if len(combined) < 300:
            raise RuntimeError(
                f"Combined Commons sources yielded only {len(combined)} mixer candidates"
            )
        return combined

    raise ValueError(f"Unsupported class: {class_name}")


builder.collect_target_candidates = collect_target_candidates


def robust_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    if not candidates:
        return []

    class_name = candidates[0].class_name
    target_valid = 260 if class_name == "cucumber" else 180
    maximum = min(max_downloads, len(candidates), 1000)
    selected = candidates[:maximum]
    result: list[base.Candidate] = []
    batch_size = 64
    actual_workers = 8 if class_name == "cucumber" else 6

    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(
                    v8.v7.v6.v5.efficient_download_one,
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
            f"[robust-download] {class_name}: "
            f"attempted={min(start + len(chunk), len(selected))}/{len(selected)}, "
            f"valid={len(result)}",
            flush=True,
        )
        if len(result) >= target_valid:
            break

    result.sort(key=base.stable_key)
    return result


base.download_candidates = robust_download_candidates


def main() -> int:
    return v8.v7.v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
