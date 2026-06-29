#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from PIL import Image

import build_dataset as base


VISIBLE_POSITIVE = {
    "feijoa": [
        "green oval feijoa fruits are clearly visible in the foreground",
        "a cut feijoa fruit with pale edible flesh is clearly visible",
        "several edible feijoa fruits occupy a substantial part of the photograph",
    ],
    "carrot": [
        "large orange carrot root vegetables are clearly visible",
        "harvested carrots occupy the foreground of the photograph",
        "a bunch of edible orange carrots is clearly visible",
    ],
    "watermelon": [
        "a whole striped watermelon fruit is clearly visible",
        "red watermelon flesh or watermelon slices are clearly visible",
        "one or more watermelon fruits occupy the foreground",
    ],
}

VISIBLE_NEGATIVE = [
    "flowers, leaves, stems, or empty ground with no edible crop visible",
    "a botanical plant photograph where the fruit or root vegetable is not visible",
    "a distant landscape with no clearly visible food item",
    "a flower close-up without any fruit or vegetable",
]


def license_is_allowed(value: str | None) -> bool:
    normalized = base.normalize_license(value)
    if not normalized:
        return False
    if normalized in {"by", "by-sa", "cc0", "pdm", "public domain", "publicdomain"}:
        return True
    if any(
        fragment in normalized
        for fragment in (
            "cc by",
            "cc-by",
            "cc by-sa",
            "cc-by-sa",
            "attribution",
            "public domain",
            "creativecommons.org/licenses/by/",
            "creativecommons.org/licenses/by-sa/",
            "creativecommons.org/publicdomain/",
        )
    ):
        return True
    return False


base.license_is_allowed = license_is_allowed


def shuffled_unique(items: list[base.Candidate], seed: int) -> list[base.Candidate]:
    result = base.unique_candidates(items)
    random.Random(seed).shuffle(result)
    return result


def collect_candidates(
    client: base.HttpClient,
    class_name: str,
    source_limit: int,
) -> list[base.Candidate]:
    config = base.CLASS_CONFIG[class_name]
    by_source: dict[str, list[base.Candidate]] = defaultdict(list)

    for query_index, query in enumerate(config["queries"]):
        print(f"[collect-v2] {class_name}: Openverse query={query!r}")
        try:
            by_source["openverse"].extend(
                base.openverse_search(client, class_name, query, source_limit)
            )
        except Exception as exc:
            print(f"[collect-v2] Openverse failed for {query!r}: {exc}", file=sys.stderr)

        print(f"[collect-v2] {class_name}: Wikimedia Commons query={query!r}")
        try:
            by_source["wikimedia_commons"].extend(
                base.commons_search(client, class_name, query, source_limit)
            )
        except Exception as exc:
            print(f"[collect-v2] Commons failed for {query!r}: {exc}", file=sys.stderr)

    scientific_name = config["scientific_name"]
    if class_name == "feijoa":
        print(f"[collect-v2] {class_name}: iNaturalist fallback={scientific_name!r}")
        try:
            by_source["inaturalist"].extend(
                base.inat_search(client, class_name, scientific_name, source_limit * 2)
            )
        except Exception as exc:
            print(f"[collect-v2] iNaturalist failed: {exc}", file=sys.stderr)
        print(f"[collect-v2] {class_name}: GBIF fallback={scientific_name!r}")
        try:
            by_source["gbif"].extend(
                base.gbif_search(client, class_name, scientific_name, source_limit)
            )
        except Exception as exc:
            print(f"[collect-v2] GBIF failed: {exc}", file=sys.stderr)

    seed_base = 2026062900 + sum(ord(char) for char in class_name)
    openverse = shuffled_unique(by_source["openverse"], seed_base + 1)
    commons = shuffled_unique(by_source["wikimedia_commons"], seed_base + 2)
    inaturalist = shuffled_unique(by_source["inaturalist"], seed_base + 3)
    gbif = shuffled_unique(by_source["gbif"], seed_base + 4)

    # Prefer web photographs whose search context names the edible object.
    # Natural-history observations are a fallback only for feijoa.
    ordered = base.unique_candidates(openverse + commons + inaturalist + gbif)
    print(
        f"[collect-v2] {class_name}: openverse={len(openverse)}, "
        f"commons={len(commons)}, inaturalist={len(inaturalist)}, "
        f"gbif={len(gbif)}, total_unique={len(ordered)}"
    )
    return ordered


base.collect_candidates = collect_candidates


def clip_score_candidates(
    candidates: list[base.Candidate],
    class_name: str,
    model: base.CLIPModel,
    processor: base.CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> list[base.Candidate]:
    config = base.CLASS_CONFIG[class_name]
    positive_features = base.clip_text_features(
        model, processor, config["positive_prompts"] + VISIBLE_POSITIVE[class_name], device
    )
    negative_features = base.clip_text_features(
        model, processor, config["negative_prompts"] + VISIBLE_NEGATIVE, device
    )
    visible_positive = base.clip_text_features(
        model, processor, VISIBLE_POSITIVE[class_name], device
    )
    visible_negative = base.clip_text_features(
        model, processor, VISIBLE_NEGATIVE, device
    )

    output: list[base.Candidate] = []
    for chunk in base.batch(candidates, batch_size):
        images: list[Image.Image] = []
        valid: list[base.Candidate] = []
        for candidate in chunk:
            try:
                with Image.open(candidate.local_path) as image:
                    image.load()
                    images.append(image.convert("RGB"))
                    valid.append(candidate)
            except Exception:
                continue
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings = features.detach().cpu().numpy().astype(np.float32)

        positive_scores = (embeddings @ positive_features.T).max(axis=1)
        negative_scores = (embeddings @ negative_features.T).max(axis=1)
        visible_margin = (
            (embeddings @ visible_positive.T).max(axis=1)
            - (embeddings @ visible_negative.T).max(axis=1)
        )
        margins = (positive_scores - negative_scores) + 0.85 * visible_margin

        for candidate, embedding, positive, negative, margin in zip(
            valid, embeddings, positive_scores, negative_scores, margins
        ):
            candidate.clip_positive = float(positive)
            candidate.clip_negative = float(negative)
            candidate.clip_margin = float(margin)
            candidate.clip_embedding = embedding.tolist()
            output.append(candidate)
        print(f"[clip-v2] {class_name}: scored {len(output)}/{len(candidates)}")
    return output


base.clip_score_candidates = clip_score_candidates


def content_filter(
    candidates: list[base.Candidate],
    minimum_needed: int,
) -> tuple[list[base.Candidate], dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.clip_margin, item.clip_positive),
        reverse=True,
    )
    if not ordered:
        raise RuntimeError("No CLIP-scored candidates")

    target_pool = max(150, minimum_needed + 40)
    thresholds = [0.08, 0.065, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    chosen: list[base.Candidate] = []
    used_threshold: float | None = None
    for threshold in thresholds:
        pool = [
            item
            for item in ordered
            if item.clip_margin >= threshold and item.clip_positive >= 0.20
        ]
        if len(pool) >= target_pool:
            chosen = pool
            used_threshold = threshold
            break

    if not chosen:
        nonnegative = [
            item
            for item in ordered
            if item.clip_margin >= 0.0 and item.clip_positive >= 0.20
        ]
        if len(nonnegative) < minimum_needed:
            raise RuntimeError(
                f"Strict object-visible filtering retained only {len(nonnegative)} "
                f"images; {minimum_needed} are required. Refusing to fill the "
                "dataset with flowers, leaves, or unrelated photographs."
            )
        chosen = nonnegative
        used_threshold = 0.0

    return chosen, {
        "threshold": used_threshold,
        "kept": len(chosen),
        "highest_margin": ordered[0].clip_margin,
        "lowest_margin_in_pool": chosen[-1].clip_margin,
        "nonnegative_total": sum(item.clip_margin >= 0.0 for item in ordered),
    }


base.content_filter = content_filter


if __name__ == "__main__":
    raise SystemExit(base.main())
