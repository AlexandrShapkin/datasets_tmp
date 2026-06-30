#!/usr/bin/env python3
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import kagglehub

import build_cucumber_mixer_open_set_v7 as v7

builder = v7.builder
base = builder.base

VEGETABLE_HANDLE = "misrakahmed/vegetable-image-dataset"
VEGETABLE_PAGE = "https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset"


def kaggle_cucumber_candidates() -> list[base.Candidate]:
    output_dir = Path(".work_open_set/vegetable_image_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        kagglehub.dataset_download(
            VEGETABLE_HANDLE,
            output_dir=str(output_dir),
        )
    )

    search_roots = [resolved, output_dir]
    files: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in (
            "**/Cucumber/*.jpg",
            "**/Cucumber/*.jpeg",
            "**/cucumber/*.jpg",
            "**/cucumber/*.jpeg",
        ):
            files.update(path.resolve() for path in root.glob(pattern) if path.is_file())

    ordered = sorted(files)
    if len(ordered) < 200:
        raise RuntimeError(
            f"Vegetable Image Dataset yielded only {len(ordered)} cucumber files"
        )
    random.Random(builder.SEED + 808).shuffle(ordered)

    candidates: list[base.Candidate] = []
    for path in ordered:
        candidates.append(
            base.Candidate(
                class_name="cucumber",
                source="kaggle_vegetable_image_dataset",
                query="Cucumber class",
                image_url=path.as_uri(),
                landing_url=VEGETABLE_PAGE,
                title=path.name,
                creator="Vegetable Image Dataset contributors",
                license="See Kaggle dataset page",
                license_url=VEGETABLE_PAGE,
                source_id=path.as_posix(),
            )
        )
    print(f"[vegetable-kaggle] cucumber files={len(candidates)}", flush=True)
    return candidates


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name == "cucumber":
        return kaggle_cucumber_candidates()
    if class_name == "concrete_mixer":
        candidates = v7.commons_html_candidates(
            class_name,
            v7.MIXER_CATEGORIES,
        )
        if len(candidates) < 180:
            raise RuntimeError(
                f"Static Commons scraper found only {len(candidates)} mixer files"
            )
        return candidates
    raise ValueError(f"Unsupported class: {class_name}")


builder.collect_target_candidates = collect_target_candidates


def expanded_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    if not candidates:
        return []

    class_name = candidates[0].class_name
    target_valid = 220
    maximum = min(max_downloads, 700)
    selected = candidates[:maximum]
    result: list[base.Candidate] = []
    batch_size = 64
    actual_workers = 8 if class_name == "cucumber" else 4

    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(
                    v7.v6.v5.efficient_download_one,
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
            f"[expanded-download] {class_name}: "
            f"attempted={min(start + len(chunk), len(selected))}/{len(selected)}, "
            f"valid={len(result)}",
            flush=True,
        )
        if len(result) >= target_valid:
            break

    result.sort(key=base.stable_key)
    return result


base.download_candidates = expanded_download_candidates


def main() -> int:
    return v7.v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
