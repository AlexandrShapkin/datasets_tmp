#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.parse
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import imagehash
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat
from transformers import CLIPModel, CLIPProcessor


USER_AGENT = (
    "AlexandrShapkin-datasets-tmp/1.0 "
    "(educational image-classification dataset builder; GitHub Actions)"
)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
INAT_API = "https://api.inaturalist.org/v1"
GBIF_API = "https://api.gbif.org/v1"
MODEL_NAME = "openai/clip-vit-base-patch32"

CLASS_CONFIG: dict[str, dict[str, Any]] = {
    "feijoa": {
        "display_name": "feijoa",
        "scientific_name": "Acca sellowiana",
        "queries": [
            "feijoa fruit",
            "feijoa fruits",
            "Acca sellowiana fruit",
            "pineapple guava fruit",
            "feijoa cut fruit",
            "feijoa market",
            "фейхоа плод",
        ],
        "positive_prompts": [
            "a clear photograph of feijoa fruit",
            "a photograph of several feijoa fruits",
            "a photograph of a cut feijoa fruit showing pale flesh",
            "a photograph of pineapple guava fruit, Acca sellowiana",
        ],
        "negative_prompts": [
            "a photograph of an avocado",
            "a photograph of a common guava",
            "a photograph of a kiwi fruit",
            "a photograph of a lime",
            "a photograph of green leaves without fruit",
            "a photograph of a flower without fruit",
            "a photograph of a tree without visible fruit",
            "a drawing or illustration of fruit",
            "a packaged food product",
        ],
    },
    "carrot": {
        "display_name": "carrot",
        "scientific_name": "Daucus carota",
        "queries": [
            "carrot vegetable",
            "fresh carrots",
            "carrot root vegetable",
            "Daucus carota root",
            "carrot harvest",
            "carrots market",
            "морковь овощ",
        ],
        "positive_prompts": [
            "a clear photograph of a carrot vegetable",
            "a photograph of fresh orange carrots",
            "a photograph of carrots with green tops",
            "a photograph of harvested carrot roots",
        ],
        "negative_prompts": [
            "a photograph of a parsnip",
            "a photograph of a sweet potato",
            "a photograph of a radish",
            "a photograph of a pumpkin",
            "a photograph of orange flowers",
            "a photograph of green leaves without carrots",
            "a drawing or illustration of a carrot",
            "a packaged food product",
        ],
    },
    "watermelon": {
        "display_name": "watermelon",
        "scientific_name": "Citrullus lanatus",
        "queries": [
            "watermelon fruit",
            "fresh watermelon",
            "Citrullus lanatus fruit",
            "watermelon slice",
            "watermelon harvest",
            "watermelons market",
            "арбуз плод",
        ],
        "positive_prompts": [
            "a clear photograph of a watermelon",
            "a photograph of whole watermelons",
            "a photograph of a cut watermelon showing red flesh",
            "a photograph of watermelon slices",
        ],
        "negative_prompts": [
            "a photograph of a cantaloupe melon",
            "a photograph of a honeydew melon",
            "a photograph of a cucumber",
            "a photograph of a pumpkin",
            "a photograph of green leaves without watermelon fruit",
            "a drawing or illustration of a watermelon",
            "a packaged food product",
        ],
    },
}

ALLOWED_LICENSE_TOKENS = {
    "cc0",
    "pdm",
    "publicdomain",
    "public domain",
    "cc by",
    "cc-by",
    "cc_by",
    "cc by-sa",
    "cc-by-sa",
    "cc_by_sa",
    "attribution",
    "attribution-sharealike",
}


@dataclass
class Candidate:
    class_name: str
    source: str
    query: str
    image_url: str
    landing_url: str
    title: str = ""
    creator: str = ""
    license: str = ""
    license_url: str = ""
    source_id: str = ""
    local_path: str = ""
    width: int = 0
    height: int = 0
    sha256: str = ""
    phash: str = ""
    dhash: str = ""
    whash: str = ""
    mirror_phash: str = ""
    clip_positive: float = 0.0
    clip_negative: float = 0.0
    clip_margin: float = 0.0
    clip_embedding: list[float] = field(default_factory=list)
    split: str = ""
    output_name: str = ""


class HttpClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: int = 35,
        stream: bool = False,
        retries: int = 4,
    ) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=timeout,
                    stream=stream,
                    allow_redirects=True,
                )
                if response.status_code == 429:
                    wait = min(30, 2 ** attempt * 3)
                    time.sleep(wait)
                    continue
                if response.status_code >= 500:
                    time.sleep(min(15, 2 ** attempt))
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                time.sleep(min(15, 2 ** attempt))
        raise RuntimeError(f"GET failed for {url}: {last_error}")


def normalize_license(value: str | None) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


def license_is_allowed(value: str | None) -> bool:
    normalized = normalize_license(value)
    if not normalized:
        return False
    return any(token in normalized for token in ALLOWED_LICENSE_TOKENS)


def stable_key(candidate: Candidate) -> str:
    return "|".join(
        [
            candidate.class_name,
            candidate.source,
            candidate.source_id,
            candidate.image_url,
            candidate.landing_url,
        ]
    )


def unique_candidates(items: Iterable[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    result: list[Candidate] = []
    for item in items:
        key = item.image_url.split("?")[0]
        fallback = stable_key(item)
        if key in seen or fallback in seen:
            continue
        seen.add(key)
        seen.add(fallback)
        result.append(item)
    return result


def commons_search(client: HttpClient, class_name: str, query: str, limit: int) -> list[Candidate]:
    page_ids: list[str] = []
    offset: str | None = None
    while len(page_ids) < limit:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srnamespace": 6,
            "srlimit": min(500, limit - len(page_ids)),
            "srsearch": f"{query} filetype:bitmap",
        }
        if offset:
            params["sroffset"] = offset
        payload = client.get(COMMONS_API, params=params).json()
        page_ids.extend(str(item["pageid"]) for item in payload.get("query", {}).get("search", []))
        continuation = payload.get("continue", {})
        offset = continuation.get("sroffset")
        if not offset:
            break

    output: list[Candidate] = []
    for start in range(0, len(page_ids), 50):
        batch_ids = page_ids[start : start + 50]
        params = {
            "action": "query",
            "format": "json",
            "pageids": "|".join(batch_ids),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 1200,
        }
        payload = client.get(COMMONS_API, params=params).json()
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/") or mime in {"image/svg+xml", "image/gif"}:
                continue
            metadata = info.get("extmetadata", {})
            license_name = (
                metadata.get("LicenseShortName", {}).get("value")
                or metadata.get("UsageTerms", {}).get("value")
                or ""
            )
            if not license_is_allowed(license_name):
                continue
            creator = metadata.get("Artist", {}).get("value", "")
            title = metadata.get("ObjectName", {}).get("value") or page.get("title", "")
            image_url = info.get("thumburl") or info.get("url")
            landing_url = info.get("descriptionurl") or (
                "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(page.get("title", "").replace(" ", "_"))
            )
            if not image_url:
                continue
            output.append(
                Candidate(
                    class_name=class_name,
                    source="wikimedia_commons",
                    query=query,
                    image_url=image_url,
                    landing_url=landing_url,
                    title=strip_html(title),
                    creator=strip_html(creator),
                    license=strip_html(license_name),
                    license_url=metadata.get("LicenseUrl", {}).get("value", ""),
                    source_id=str(page.get("pageid", "")),
                )
            )
    return output


def openverse_search(client: HttpClient, class_name: str, query: str, limit: int) -> list[Candidate]:
    output: list[Candidate] = []
    page = 1
    page_size = min(80, limit)
    while len(output) < limit and page <= 8:
        params = {
            "q": query,
            "page_size": page_size,
            "page": page,
            "license": "cc0,pdm,by,by-sa",
            "mature": "false",
        }
        try:
            payload = client.get(OPENVERSE_API, params=params, timeout=45).json()
        except Exception as exc:
            print(f"[openverse] query={query!r} page={page}: {exc}", file=sys.stderr)
            break
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            license_name = item.get("license", "")
            if not license_is_allowed(license_name):
                continue
            image_url = item.get("thumbnail") or item.get("url")
            landing_url = item.get("foreign_landing_url") or item.get("detail_url") or item.get("url")
            if not image_url or not landing_url:
                continue
            output.append(
                Candidate(
                    class_name=class_name,
                    source="openverse",
                    query=query,
                    image_url=image_url,
                    landing_url=landing_url,
                    title=item.get("title") or "",
                    creator=item.get("creator") or "",
                    license=license_name,
                    license_url=item.get("license_url") or "",
                    source_id=str(item.get("id") or ""),
                )
            )
            if len(output) >= limit:
                break
        page += 1
    return output


def inat_taxon_id(client: HttpClient, scientific_name: str) -> int | None:
    payload = client.get(
        f"{INAT_API}/taxa",
        params={"q": scientific_name, "rank": "species", "per_page": 20},
    ).json()
    exact = None
    fallback = None
    for item in payload.get("results", []):
        if fallback is None:
            fallback = item.get("id")
        if (item.get("name") or "").lower() == scientific_name.lower():
            exact = item.get("id")
            break
    return int(exact or fallback) if (exact or fallback) else None


def inat_search(
    client: HttpClient,
    class_name: str,
    scientific_name: str,
    limit: int,
) -> list[Candidate]:
    taxon_id = inat_taxon_id(client, scientific_name)
    if not taxon_id:
        return []
    output: list[Candidate] = []
    pages = max(1, math.ceil(limit / 200))
    for page in range(1, min(pages, 6) + 1):
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "per_page": 200,
            "page": page,
            "order_by": "created_at",
            "order": "desc",
        }
        payload = client.get(f"{INAT_API}/observations", params=params, timeout=50).json()
        results = payload.get("results", [])
        if not results:
            break
        for observation in results:
            landing = observation.get("uri") or f"https://www.inaturalist.org/observations/{observation.get('id')}"
            for photo in observation.get("photos", []):
                license_name = photo.get("license_code") or ""
                if not license_is_allowed(license_name):
                    continue
                url = photo.get("url") or ""
                if not url:
                    continue
                image_url = url.replace("/square.", "/large.").replace("/small.", "/large.")
                output.append(
                    Candidate(
                        class_name=class_name,
                        source="inaturalist",
                        query=scientific_name,
                        image_url=image_url,
                        landing_url=landing,
                        title=observation.get("species_guess") or scientific_name,
                        creator=photo.get("attribution") or observation.get("user", {}).get("login", ""),
                        license=license_name,
                        license_url="https://creativecommons.org/licenses/",
                        source_id=f"{observation.get('id')}:{photo.get('id')}",
                    )
                )
                if len(output) >= limit:
                    return output
    return output


def gbif_species_key(client: HttpClient, scientific_name: str) -> int | None:
    payload = client.get(f"{GBIF_API}/species/match", params={"name": scientific_name}).json()
    key = payload.get("usageKey") or payload.get("speciesKey")
    return int(key) if key else None


def gbif_search(
    client: HttpClient,
    class_name: str,
    scientific_name: str,
    limit: int,
) -> list[Candidate]:
    key = gbif_species_key(client, scientific_name)
    if not key:
        return []
    output: list[Candidate] = []
    offset = 0
    while len(output) < limit and offset < 1000:
        params = {
            "taxon_key": key,
            "media_type": "StillImage",
            "limit": min(300, limit - len(output)),
            "offset": offset,
        }
        payload = client.get(f"{GBIF_API}/occurrence/search", params=params, timeout=50).json()
        results = payload.get("results", [])
        if not results:
            break
        for occurrence in results:
            landing = (
                occurrence.get("references")
                or f"https://www.gbif.org/occurrence/{occurrence.get('key')}"
            )
            for media in occurrence.get("media", []):
                image_url = media.get("identifier") or media.get("references")
                if not image_url or not str(image_url).startswith("http"):
                    continue
                license_name = media.get("license") or occurrence.get("license") or ""
                if not license_is_allowed(license_name):
                    continue
                output.append(
                    Candidate(
                        class_name=class_name,
                        source="gbif",
                        query=scientific_name,
                        image_url=image_url,
                        landing_url=landing,
                        title=media.get("title") or occurrence.get("scientificName") or scientific_name,
                        creator=media.get("creator") or occurrence.get("recordedBy") or "",
                        license=license_name,
                        license_url=media.get("license") or "",
                        source_id=f"{occurrence.get('key')}:{media.get('identifier', '')}",
                    )
                )
                if len(output) >= limit:
                    return output
        offset += len(results)
        if payload.get("endOfRecords"):
            break
    return output


def strip_html(value: Any) -> str:
    text = str(value or "")
    inside = False
    result: list[str] = []
    for char in text:
        if char == "<":
            inside = True
            continue
        if char == ">":
            inside = False
            continue
        if not inside:
            result.append(char)
    return " ".join("".join(result).replace("&nbsp;", " ").split())


def collect_candidates(
    client: HttpClient,
    class_name: str,
    source_limit: int,
) -> list[Candidate]:
    config = CLASS_CONFIG[class_name]
    all_items: list[Candidate] = []

    for query in config["queries"]:
        print(f"[collect] {class_name}: Wikimedia Commons query={query!r}")
        try:
            all_items.extend(commons_search(client, class_name, query, source_limit))
        except Exception as exc:
            print(f"[collect] Commons failed for {query!r}: {exc}", file=sys.stderr)

        print(f"[collect] {class_name}: Openverse query={query!r}")
        try:
            all_items.extend(openverse_search(client, class_name, query, source_limit))
        except Exception as exc:
            print(f"[collect] Openverse failed for {query!r}: {exc}", file=sys.stderr)

    scientific_name = config["scientific_name"]
    print(f"[collect] {class_name}: iNaturalist taxon={scientific_name!r}")
    try:
        all_items.extend(inat_search(client, class_name, scientific_name, source_limit * 3))
    except Exception as exc:
        print(f"[collect] iNaturalist failed for {class_name}: {exc}", file=sys.stderr)

    print(f"[collect] {class_name}: GBIF taxon={scientific_name!r}")
    try:
        all_items.extend(gbif_search(client, class_name, scientific_name, source_limit * 2))
    except Exception as exc:
        print(f"[collect] GBIF failed for {class_name}: {exc}", file=sys.stderr)

    unique = unique_candidates(all_items)
    random.Random(4815162342 + len(class_name)).shuffle(unique)
    print(
        f"[collect] {class_name}: {len(all_items)} raw candidates, "
        f"{len(unique)} unique URLs"
    )
    return unique


def image_entropy(image: Image.Image) -> float:
    gray = image.convert("L").resize((128, 128))
    histogram = np.asarray(gray.histogram(), dtype=np.float64)
    probability = histogram / max(histogram.sum(), 1.0)
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def decode_image(content: bytes) -> Image.Image:
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def prepare_image(image: Image.Image, max_side: int = 1280) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def download_one(
    candidate: Candidate,
    download_root: Path,
    index: int,
) -> Candidate | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"}
    try:
        response = requests.get(
            candidate.image_url,
            headers=headers,
            timeout=40,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        content = response.content
        if len(content) < 8_000 or len(content) > 30_000_000:
            return None
        if content_type and "image" not in content_type.lower():
            if "octet-stream" not in content_type.lower():
                return None
        image = decode_image(content)
        width, height = image.size
        if min(width, height) < 180:
            return None
        if max(width, height) / max(1, min(width, height)) > 4.5:
            return None
        if image_entropy(image) < 3.4:
            return None
        stat = ImageStat.Stat(image.resize((64, 64)))
        if max(stat.var) < 25:
            return None

        image = prepare_image(image)
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
        candidate.mirror_phash = str(imagehash.phash(ImageOps.mirror(image), hash_size=16))
        return candidate
    except Exception:
        return None


def download_candidates(
    candidates: list[Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[Candidate]:
    selected = candidates[:max_downloads]
    result: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, candidate, download_root, index): candidate
            for index, candidate in enumerate(selected)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            item = future.result()
            if item:
                result.append(item)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"[download] completed={completed}/{len(futures)} "
                    f"valid={len(result)}"
                )
    result.sort(key=stable_key)
    return result


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def preclip_deduplicate(candidates: list[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    kept: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    exact_seen: dict[str, Candidate] = {}

    for candidate in candidates:
        if candidate.sha256 in exact_seen:
            rejected.append(
                {
                    "reason": "exact_sha256",
                    "candidate": stable_key(candidate),
                    "duplicate_of": stable_key(exact_seen[candidate.sha256]),
                }
            )
            continue

        duplicate_of: Candidate | None = None
        duplicate_reason = ""
        for existing in kept:
            phash_distance = min(
                hamming_hex(candidate.phash, existing.phash),
                hamming_hex(candidate.mirror_phash, existing.phash),
                hamming_hex(candidate.phash, existing.mirror_phash),
            )
            dhash_distance = hamming_hex(candidate.dhash, existing.dhash)
            whash_distance = hamming_hex(candidate.whash, existing.whash)

            if phash_distance <= 6 and dhash_distance <= 10:
                duplicate_of = existing
                duplicate_reason = (
                    f"perceptual_hash:phash={phash_distance},"
                    f"dhash={dhash_distance},whash={whash_distance}"
                )
                break
            if phash_distance <= 9 and dhash_distance <= 8 and whash_distance <= 10:
                duplicate_of = existing
                duplicate_reason = (
                    f"perceptual_hash_combined:phash={phash_distance},"
                    f"dhash={dhash_distance},whash={whash_distance}"
                )
                break

        if duplicate_of:
            rejected.append(
                {
                    "reason": duplicate_reason,
                    "candidate": stable_key(candidate),
                    "duplicate_of": stable_key(duplicate_of),
                }
            )
            continue

        exact_seen[candidate.sha256] = candidate
        kept.append(candidate)

    return kept, rejected


def batch(iterable: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(iterable), size):
        yield iterable[index : index + size]


def clip_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompts: list[str],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().numpy().astype(np.float32)


def clip_score_candidates(
    candidates: list[Candidate],
    class_name: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> list[Candidate]:
    config = CLASS_CONFIG[class_name]
    positive_features = clip_text_features(
        model, processor, config["positive_prompts"], device
    )
    negative_features = clip_text_features(
        model, processor, config["negative_prompts"], device
    )
    positive_vector = positive_features.mean(axis=0)
    positive_vector /= np.linalg.norm(positive_vector) + 1e-12

    output: list[Candidate] = []
    for chunk in batch(candidates, batch_size):
        images: list[Image.Image] = []
        valid: list[Candidate] = []
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
        positive_scores = embeddings @ positive_vector
        negative_scores = (embeddings @ negative_features.T).max(axis=1)
        margins = positive_scores - negative_scores

        for candidate, embedding, positive, negative, margin in zip(
            valid, embeddings, positive_scores, negative_scores, margins
        ):
            candidate.clip_positive = float(positive)
            candidate.clip_negative = float(negative)
            candidate.clip_margin = float(margin)
            candidate.clip_embedding = embedding.tolist()
            output.append(candidate)
        print(f"[clip] {class_name}: scored {len(output)}/{len(candidates)}")
    return output


def content_filter(candidates: list[Candidate], minimum_needed: int) -> tuple[list[Candidate], dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.clip_margin, item.clip_positive),
        reverse=True,
    )
    if not ordered:
        return [], {"threshold": None, "kept": 0}

    thresholds = [0.025, 0.015, 0.005, -0.005, -0.02, -0.04]
    target_pool = max(minimum_needed * 2, minimum_needed + 80)
    chosen = ordered
    used_threshold = thresholds[-1]
    for threshold in thresholds:
        pool = [
            item
            for item in ordered
            if item.clip_margin >= threshold and item.clip_positive >= 0.18
        ]
        if len(pool) >= target_pool:
            chosen = pool
            used_threshold = threshold
            break
    else:
        chosen = ordered[: max(target_pool, minimum_needed)]
        used_threshold = chosen[-1].clip_margin if chosen else None

    return chosen, {
        "threshold": used_threshold,
        "kept": len(chosen),
        "highest_margin": ordered[0].clip_margin,
        "lowest_margin_in_pool": chosen[-1].clip_margin if chosen else None,
    }


def embedding_matrix(candidates: list[Candidate]) -> np.ndarray:
    matrix = np.asarray([item.clip_embedding for item in candidates], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embedding_deduplicate(
    candidates: list[Candidate],
    cosine_threshold: float = 0.992,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.clip_margin, item.clip_positive),
        reverse=True,
    )
    kept: list[Candidate] = []
    kept_vectors: list[np.ndarray] = []
    rejected: list[dict[str, Any]] = []

    for candidate in ordered:
        vector = np.asarray(candidate.clip_embedding, dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        duplicate_index = None
        duplicate_similarity = None
        if kept_vectors:
            similarities = np.asarray(kept_vectors) @ vector
            index = int(similarities.argmax())
            similarity = float(similarities[index])
            if similarity >= cosine_threshold:
                duplicate_index = index
                duplicate_similarity = similarity

        if duplicate_index is not None:
            rejected.append(
                {
                    "reason": "clip_near_duplicate",
                    "similarity": duplicate_similarity,
                    "candidate": stable_key(candidate),
                    "duplicate_of": stable_key(kept[duplicate_index]),
                }
            )
            continue
        kept.append(candidate)
        kept_vectors.append(vector)

    return kept, rejected


def diverse_select(candidates: list[Candidate], count: int) -> list[Candidate]:
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} candidates remain, but {count} are required"
        )
    matrix = embedding_matrix(candidates)
    margins = np.asarray([item.clip_margin for item in candidates], dtype=np.float32)
    positives = np.asarray([item.clip_positive for item in candidates], dtype=np.float32)

    def normalize(values: np.ndarray) -> np.ndarray:
        low = float(np.percentile(values, 5))
        high = float(np.percentile(values, 95))
        if high <= low:
            return np.ones_like(values)
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    quality = 0.7 * normalize(margins) + 0.3 * normalize(positives)
    first = int(np.argmax(quality))
    selected = [first]
    available = np.ones(len(candidates), dtype=bool)
    available[first] = False
    min_distance = 1.0 - matrix @ matrix[first]

    while len(selected) < count:
        objective = 0.62 * quality + 0.38 * np.clip(min_distance / 0.35, 0.0, 1.0)
        objective[~available] = -np.inf
        index = int(np.argmax(objective))
        if not np.isfinite(objective[index]):
            break
        selected.append(index)
        available[index] = False
        distance = 1.0 - matrix @ matrix[index]
        min_distance = np.minimum(min_distance, distance)

    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of {count} images")
    return [candidates[index] for index in selected]


def assign_splits(candidates: list[Candidate], seed: int) -> None:
    random.Random(seed).shuffle(candidates)
    total = len(candidates)
    train_count = round(total * 0.70)
    val_count = round(total * 0.15)
    for index, candidate in enumerate(candidates):
        if index < train_count:
            candidate.split = "train"
        elif index < train_count + val_count:
            candidate.split = "val"
        else:
            candidate.split = "test"


def save_final_image(source_path: Path, destination: Path) -> tuple[int, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = prepare_image(image, max_side=1280)
        width, height = image.size
        image.save(destination, "JPEG", quality=94, optimize=True, subsampling=0)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return width, height, digest


def make_contact_sheet(
    candidates: list[Candidate],
    output_path: Path,
    title: str,
    columns: int = 10,
    cell_size: int = 180,
) -> None:
    rows = math.ceil(len(candidates) / columns)
    header = 60
    sheet = Image.new("RGB", (columns * cell_size, header + rows * cell_size), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)
    draw.text(
        (12, 32),
        "Each tile is one independent source photograph; no augmentation copies.",
        fill="black",
        font=font,
    )

    for index, candidate in enumerate(candidates):
        row, column = divmod(index, columns)
        left = column * cell_size
        top = header + row * cell_size
        with Image.open(candidate.local_path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            thumbnail = ImageOps.contain(
                image,
                (cell_size - 8, cell_size - 28),
                Image.Resampling.LANCZOS,
            )
        x = left + (cell_size - thumbnail.width) // 2
        y = top + 2
        sheet.paste(thumbnail, (x, y))
        draw.rectangle(
            (left, top, left + cell_size - 1, top + cell_size - 1),
            outline="gray",
            width=1,
        )
        label = f"{index + 1:03d} {candidate.source[:9]} m={candidate.clip_margin:+.3f}"
        draw.text((left + 4, top + cell_size - 22), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "JPEG", quality=90, optimize=True)


def nearest_pairs(candidates: list[Candidate], limit: int = 20) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return []
    matrix = embedding_matrix(candidates)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)
    pairs: list[tuple[float, int, int]] = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            pairs.append((float(similarity[left, right]), left, right))
    pairs.sort(reverse=True)
    return [
        {
            "cosine_similarity": score,
            "left": candidates[left].output_name,
            "right": candidates[right].output_name,
        }
        for score, left, right in pairs[:limit]
    ]


def build_readme(count: int, split_counts: dict[str, dict[str, int]]) -> str:
    return f"""# Feijoa, carrot and watermelon classification dataset

This archive contains **{count} independently sourced photographs per class**.

Classes:

- `feijoa`
- `carrot`
- `watermelon`

No rotation, reflection, color edit, crop variation, or other augmentation has
been counted as an additional source image. Training augmentation should be
performed dynamically in the model's data loader.

## Directory layout

```text
dataset/
├── train/
│   ├── feijoa/
│   ├── carrot/
│   └── watermelon/
├── val/
│   ├── feijoa/
│   ├── carrot/
│   └── watermelon/
├── test/
│   ├── feijoa/
│   ├── carrot/
│   └── watermelon/
├── manifest.csv
├── validation.json
├── duplicate_rejections.json
├── contact_sheets/
└── LICENSE_NOTICE.md
```

Expected split per class:

- train: {split_counts["feijoa"]["train"]}
- val: {split_counts["feijoa"]["val"]}
- test: {split_counts["feijoa"]["test"]}

## Quality controls

1. Invalid, tiny, extreme-aspect-ratio, blank, and corrupt images are rejected.
2. Exact duplicates are removed by SHA-256.
3. Recompressed, resized, and mirrored duplicates are removed with perceptual
   hashes.
4. CLIP ranks photographs against positive and confusing negative concepts.
5. CLIP embedding similarity removes remaining near-duplicates.
6. A diversity-aware selector avoids choosing a set made only of nearly
   identical studio photographs.
7. Contact sheets allow rapid visual review of all final images.

## Sources and licenses

Images come from Wikimedia Commons, Openverse-indexed collections,
iNaturalist, and GBIF. Only records carrying a public-domain, CC0, CC BY, or
CC BY-SA style license are accepted by the collector.

`manifest.csv` records the source page, creator information supplied by the
source, license string, download URL, perceptual hashes, and CLIP scores for
every image.

Always inspect the source page before republishing the dataset, because source
metadata can be incomplete or corrected after collection.
"""


def validate_dataset(
    selected_by_class: dict[str, list[Candidate]],
    dataset_root: Path,
    required_count: int,
) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    all_sha: list[str] = []

    for class_name, selected in selected_by_class.items():
        counts[class_name] = {}
        if len(selected) != required_count:
            errors.append(
                f"{class_name}: expected {required_count} selected images, got {len(selected)}"
            )
        for split in ("train", "val", "test"):
            files = sorted((dataset_root / split / class_name).glob("*.jpg"))
            counts[class_name][split] = len(files)
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB":
                            errors.append(f"{path}: image mode is {image.mode}, expected RGB")
                        if min(image.size) < 180:
                            errors.append(f"{path}: image too small: {image.size}")
                    all_sha.append(hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception as exc:
                    errors.append(f"{path}: unreadable image: {exc}")

    if len(all_sha) != len(set(all_sha)):
        errors.append("Exact duplicate files detected in final dataset")

    source_keys: list[str] = []
    for selected in selected_by_class.values():
        source_keys.extend(stable_key(item) for item in selected)
    if len(source_keys) != len(set(source_keys)):
        errors.append("A source record is reused more than once")

    return {
        "status": "ok" if not errors else "failed",
        "required_per_class": required_count,
        "total_images": len(all_sha),
        "unique_sha256": len(set(all_sha)),
        "counts": counts,
        "errors": errors,
    }


def write_manifest(candidates: list[Candidate], path: Path) -> None:
    fields = [
        "relative_path",
        "class",
        "split",
        "source",
        "source_id",
        "query",
        "title",
        "creator",
        "license",
        "license_url",
        "landing_url",
        "download_url",
        "width",
        "height",
        "sha256",
        "phash",
        "dhash",
        "whash",
        "mirror_phash",
        "clip_positive",
        "clip_negative",
        "clip_margin",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in sorted(candidates, key=lambda candidate: (candidate.class_name, candidate.split, candidate.output_name)):
            writer.writerow(
                {
                    "relative_path": f"{item.split}/{item.class_name}/{item.output_name}",
                    "class": item.class_name,
                    "split": item.split,
                    "source": item.source,
                    "source_id": item.source_id,
                    "query": item.query,
                    "title": item.title,
                    "creator": item.creator,
                    "license": item.license,
                    "license_url": item.license_url,
                    "landing_url": item.landing_url,
                    "download_url": item.image_url,
                    "width": item.width,
                    "height": item.height,
                    "sha256": item.sha256,
                    "phash": item.phash,
                    "dhash": item.dhash,
                    "whash": item.whash,
                    "mirror_phash": item.mirror_phash,
                    "clip_positive": f"{item.clip_positive:.8f}",
                    "clip_negative": f"{item.clip_negative:.8f}",
                    "clip_margin": f"{item.clip_margin:.8f}",
                }
            )


def write_checksums(dataset_root: Path) -> None:
    lines: list[str] = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(dataset_root).as_posix()}")
    (dataset_root / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def zip_directory(source_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                arcname = (source_root.name / path.relative_to(source_root)).as_posix()
                archive.write(path, arcname=arcname)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=900)
    parser.add_argument("--source-limit", type=int, default=180)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--work", type=Path, default=Path(".work"))
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.work.exists():
        shutil.rmtree(args.work)
    if args.output.exists():
        shutil.rmtree(args.output)
    args.work.mkdir(parents=True)
    args.output.mkdir(parents=True)

    downloads_root = args.work / "downloads"
    dataset_root = args.output / "dataset"
    dataset_root.mkdir(parents=True)

    client = HttpClient()
    collected_by_class: dict[str, list[Candidate]] = {}
    downloaded_by_class: dict[str, list[Candidate]] = {}
    preclip_rejections: dict[str, list[dict[str, Any]]] = {}

    for class_name in CLASS_CONFIG:
        collected = collect_candidates(client, class_name, args.source_limit)
        collected_by_class[class_name] = collected
        downloaded = download_candidates(
            collected,
            downloads_root,
            max_downloads=args.candidate_limit,
            workers=args.workers,
        )
        deduplicated, rejected = preclip_deduplicate(downloaded)
        downloaded_by_class[class_name] = deduplicated
        preclip_rejections[class_name] = rejected
        print(
            f"[preclip] {class_name}: downloaded={len(downloaded)}, "
            f"unique={len(deduplicated)}, rejected={len(rejected)}"
        )
        if len(deduplicated) < args.count:
            raise RuntimeError(
                f"{class_name}: only {len(deduplicated)} unique valid images "
                f"after download; need {args.count}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] loading {MODEL_NAME} on {device}")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    selected_by_class: dict[str, list[Candidate]] = {}
    all_rejections: dict[str, Any] = {}
    class_reports: dict[str, Any] = {}

    for class_index, class_name in enumerate(CLASS_CONFIG):
        scored = clip_score_candidates(
            downloaded_by_class[class_name],
            class_name,
            model,
            processor,
            device,
            args.batch_size,
        )
        filtered, filter_report = content_filter(scored, args.count)
        embedding_unique, embedding_rejected = embedding_deduplicate(filtered)
        if len(embedding_unique) < args.count:
            embedding_unique, embedding_rejected = embedding_deduplicate(
                filtered,
                cosine_threshold=0.996,
            )
        selected = diverse_select(embedding_unique, args.count)
        assign_splits(selected, seed=args.seed + class_index * 1009)
        selected_by_class[class_name] = selected

        all_rejections[class_name] = {
            "preclip": preclip_rejections[class_name],
            "embedding": embedding_rejected,
        }
        class_reports[class_name] = {
            "raw_collected_urls": len(collected_by_class[class_name]),
            "downloaded_and_valid": len(downloaded_by_class[class_name]) + len(preclip_rejections[class_name]),
            "after_hash_deduplication": len(downloaded_by_class[class_name]),
            "content_filter": filter_report,
            "after_embedding_deduplication": len(embedding_unique),
            "selected": len(selected),
            "source_distribution": dict(Counter(item.source for item in selected)),
            "minimum_clip_margin": min(item.clip_margin for item in selected),
            "median_clip_margin": float(np.median([item.clip_margin for item in selected])),
        }
        print(
            f"[select] {class_name}: filtered={len(filtered)}, "
            f"embedding_unique={len(embedding_unique)}, selected={len(selected)}"
        )

    final_candidates: list[Candidate] = []
    for class_name, selected in selected_by_class.items():
        ordered = sorted(
            selected,
            key=lambda item: (
                {"train": 0, "val": 1, "test": 2}[item.split],
                -item.clip_margin,
                stable_key(item),
            ),
        )
        for index, candidate in enumerate(ordered, start=1):
            candidate.output_name = f"{class_name}_{index:03d}.jpg"
            destination = dataset_root / candidate.split / class_name / candidate.output_name
            width, height, digest = save_final_image(
                Path(candidate.local_path),
                destination,
            )
            candidate.width = width
            candidate.height = height
            candidate.sha256 = digest
            final_candidates.append(candidate)

        make_contact_sheet(
            sorted(selected, key=lambda item: item.output_name),
            dataset_root / "contact_sheets" / f"{class_name}.jpg",
            title=f"{class_name}: {len(selected)} unique source photographs",
        )

    split_counts = {
        class_name: dict(Counter(item.split for item in selected))
        for class_name, selected in selected_by_class.items()
    }

    write_manifest(final_candidates, dataset_root / "manifest.csv")
    (dataset_root / "README.md").write_text(
        build_readme(args.count, split_counts),
        encoding="utf-8",
    )
    (dataset_root / "LICENSE_NOTICE.md").write_text(
        """# License notice

The builder accepts only source records whose metadata declares a public-domain,
CC0, CC BY, or CC BY-SA style license. License and attribution fields are
copied into `manifest.csv`.

Source metadata can be incomplete or change over time. Anyone redistributing
the archive must review each `landing_url` and comply with the exact attribution
and share-alike requirements shown there.
""",
        encoding="utf-8",
    )
    (dataset_root / "duplicate_rejections.json").write_text(
        json.dumps(all_rejections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_dataset(selected_by_class, dataset_root, args.count)
    validation["class_reports"] = class_reports
    validation["nearest_pairs"] = {
        class_name: nearest_pairs(selected)
        for class_name, selected in selected_by_class.items()
    }
    (dataset_root / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if validation["errors"]:
        raise RuntimeError("\n".join(validation["errors"]))

    write_checksums(dataset_root)

    zip_path = args.output / "feijoa_carrot_watermelon_unique_classification.zip"
    zip_directory(dataset_root, zip_path)
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    build_summary = {
        "archive": zip_path.name,
        "archive_bytes": zip_path.stat().st_size,
        "archive_sha256": zip_sha,
        "total_images": validation["total_images"],
        "counts": validation["counts"],
        "class_reports": class_reports,
    }
    (args.output / "build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(build_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
