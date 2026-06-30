#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import build_cucumber_mixer_open_set_v3 as v3

builder = v3.builder
base = builder.base

ORIGINAL_COLLECT = v3.ORIGINAL_COLLECT


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name != "cucumber":
        candidates = ORIGINAL_COLLECT(class_name)
        # Prefer Wikimedia records because their direct image URLs and licensing
        # metadata are generally more stable than aggregator thumbnails.
        return sorted(
            candidates,
            key=lambda item: (item.source != "wikimedia_commons", base.stable_key(item)),
        )

    kaggle = v3.kaggle_cucumber_candidates()
    web = ORIGINAL_COLLECT("cucumber")
    web = sorted(
        web,
        key=lambda item: (item.source != "wikimedia_commons", base.stable_key(item)),
    )[:280]
    combined = base.unique_candidates(kaggle + web)
    print(
        f"[combined-cucumber] kaggle={len(kaggle)}, web_reserve={len(web)}, "
        f"combined={len(combined)}",
        flush=True,
    )
    return combined


builder.collect_target_candidates = collect_target_candidates


def adaptive_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    if not candidates:
        return []
    class_name = candidates[0].class_name
    target_valid = 135 if class_name == "cucumber" else 180
    maximum = min(max_downloads, 520 if class_name == "concrete_mixer" else 360)
    selected = candidates[:maximum]
    result: list[base.Candidate] = []
    batch_size = 64

    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
            futures = {
                executor.submit(
                    v3.local_file_download,
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
            f"[adaptive-download] {class_name}: attempted={min(start + len(chunk), len(selected))}/"
            f"{len(selected)}, valid={len(result)}",
            flush=True,
        )
        if len(result) >= target_valid:
            break

    result.sort(key=base.stable_key)
    return result


base.download_candidates = adaptive_download_candidates


if __name__ == "__main__":
    raise SystemExit(builder.main())
