#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import build_dataset_v2 as v2

base = v2.base
ORIGINAL_CONTACT_SHEET = base.make_contact_sheet
REVIEW_POOLS: dict[str, list[base.Candidate]] = {}

FLOWER_TERMS = {
    "flower",
    "flowers",
    "flowering",
    "blossom",
    "blossoms",
    "bloom",
    "inflorescence",
    "umbel",
}
LEAF_TERMS = {"leaf", "leaves", "foliage", "seedling", "sprout"}
OBJECT_TERMS = {
    "fruit",
    "fruits",
    "carrot",
    "carrots",
    "watermelon",
    "watermelons",
    "feijoa",
    "harvest",
    "slice",
    "sliced",
    "cut",
    "market",
}
SOURCE_BONUS = {
    "openverse": 0.045,
    "wikimedia_commons": 0.025,
    "inaturalist": 0.0,
    "gbif": -0.005,
}


def metadata_adjustment(candidate: base.Candidate) -> float:
    text = f"{candidate.title} {candidate.query}".lower()
    words = set(text.replace("-", " ").replace("_", " ").split())
    adjustment = SOURCE_BONUS.get(candidate.source, 0.0)
    if words & FLOWER_TERMS:
        adjustment -= 0.16
    if words & LEAF_TERMS and not words & OBJECT_TERMS:
        adjustment -= 0.08
    if words & OBJECT_TERMS:
        adjustment += 0.018
    return adjustment


def content_filter(
    candidates: list[base.Candidate],
    minimum_needed: int,
) -> tuple[list[base.Candidate], dict[str, Any]]:
    usable: list[base.Candidate] = []
    for candidate in candidates:
        candidate.clip_margin += metadata_adjustment(candidate)
        if candidate.clip_positive < 0.18:
            continue
        if candidate.clip_margin < -0.08:
            continue
        usable.append(candidate)

    ordered = sorted(
        usable,
        key=lambda item: (item.clip_margin, item.clip_positive),
        reverse=True,
    )
    if len(ordered) < minimum_needed:
        raise RuntimeError(
            f"Only {len(ordered)} metadata- and CLIP-filtered candidates remain; "
            f"{minimum_needed} are required"
        )

    pool_size = min(len(ordered), max(240, minimum_needed * 2))
    pool = ordered[:pool_size]
    REVIEW_POOLS[pool[0].class_name] = pool
    return pool, {
        "threshold": pool[-1].clip_margin,
        "kept": len(pool),
        "highest_margin": pool[0].clip_margin,
        "lowest_margin_in_pool": pool[-1].clip_margin,
        "review_pool_size": len(pool),
    }


base.content_filter = content_filter


def make_contact_sheet(
    candidates: list[base.Candidate],
    output_path: Path,
    title: str,
    columns: int = 10,
    cell_size: int = 180,
) -> None:
    ORIGINAL_CONTACT_SHEET(
        candidates,
        output_path,
        title,
        columns=columns,
        cell_size=cell_size,
    )
    if not candidates:
        return
    class_name = candidates[0].class_name
    pool = REVIEW_POOLS.get(class_name, [])
    if not pool:
        return

    dataset_root = output_path.parent.parent
    pool_dir = dataset_root / "review_pool" / class_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(pool, start=1):
        filename = f"{class_name}_pool_{index:03d}.jpg"
        shutil.copy2(candidate.local_path, pool_dir / filename)
        rows.append(
            {
                "filename": filename,
                "source": candidate.source,
                "source_id": candidate.source_id,
                "title": candidate.title,
                "creator": candidate.creator,
                "license": candidate.license,
                "license_url": candidate.license_url,
                "landing_url": candidate.landing_url,
                "download_url": candidate.image_url,
                "clip_positive": candidate.clip_positive,
                "clip_negative": candidate.clip_negative,
                "adjusted_margin": candidate.clip_margin,
                "phash": candidate.phash,
                "dhash": candidate.dhash,
                "whash": candidate.whash,
                "mirror_phash": candidate.mirror_phash,
            }
        )

    manifest_path = dataset_root / "review_pool" / f"{class_name}_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ORIGINAL_CONTACT_SHEET(
        pool,
        output_path.parent / f"{class_name}_review_pool.jpg",
        f"{class_name}: review pool ({len(pool)} candidates)",
        columns=12,
        cell_size=150,
    )


base.make_contact_sheet = make_contact_sheet


if __name__ == "__main__":
    raise SystemExit(base.main())
