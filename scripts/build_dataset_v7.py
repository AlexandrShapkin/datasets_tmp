#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import build_dataset_v6 as v6

base = v6.base


def content_filter(
    candidates: list[base.Candidate],
    minimum_needed: int,
) -> tuple[list[base.Candidate], dict[str, Any]]:
    if not candidates or candidates[0].class_name != "feijoa":
        return v6.ORIGINAL_CONTENT_FILTER(candidates, minimum_needed)

    eligible: list[base.Candidate] = []
    rejected_reasons: dict[str, int] = {
        "metadata_flower": 0,
        "red_or_pink": 0,
        "insufficient_green": 0,
        "clip_fruit_score": 0,
        "very_strong_flower_margin": 0,
    }

    for candidate in candidates:
        if v6.metadata_contains_flower(candidate):
            rejected_reasons["metadata_flower"] += 1
            continue

        red_fraction = float(getattr(candidate, "feijoa_red_fraction", 1.0))
        green_fraction = float(getattr(candidate, "feijoa_green_fraction", 0.0))
        raw_margin = float(getattr(candidate, "feijoa_raw_margin", -1.0))

        # Keep the strict visual flower rejection from v6. Only the CLIP
        # margin is relaxed because it also rejects distant fruit photographs.
        if red_fraction > 0.055:
            rejected_reasons["red_or_pink"] += 1
            continue
        if green_fraction < 0.025:
            rejected_reasons["insufficient_green"] += 1
            continue
        if candidate.clip_positive < 0.205:
            rejected_reasons["clip_fruit_score"] += 1
            continue
        if raw_margin < -0.055:
            rejected_reasons["very_strong_flower_margin"] += 1
            continue

        eligible.append(candidate)

    eligible.sort(
        key=lambda item: (
            float(getattr(item, "feijoa_raw_margin", -1.0)),
            item.clip_positive,
            float(getattr(item, "feijoa_green_fraction", 0.0)),
            -float(getattr(item, "feijoa_red_fraction", 1.0)),
        ),
        reverse=True,
    )

    if len(eligible) < minimum_needed:
        raise RuntimeError(
            "Fruit-only feijoa filtering retained "
            f"{len(eligible)} images; {minimum_needed} are required. "
            f"Rejected counts: {rejected_reasons}."
        )

    pool_size = min(len(eligible), max(200, minimum_needed * 2))
    pool = eligible[:pool_size]
    v6.v5.v3.REVIEW_POOLS["feijoa"] = pool

    return pool, {
        "policy": "fruit_only_color_strict_clip_ranked",
        "kept": len(pool),
        "eligible": len(eligible),
        "rejected_reasons": rejected_reasons,
        "minimum_raw_margin_in_pool": min(
            float(getattr(item, "feijoa_raw_margin", 0.0)) for item in pool
        ),
        "maximum_red_fraction_in_pool": max(
            float(getattr(item, "feijoa_red_fraction", 0.0)) for item in pool
        ),
    }


base.content_filter = content_filter


if __name__ == "__main__":
    raise SystemExit(base.main())
