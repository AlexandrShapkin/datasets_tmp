#!/usr/bin/env python3
from __future__ import annotations

import random
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

import requests

import build_cucumber_mixer_open_set_v7 as v7

builder = v7.builder
base = builder.base

COMMONS_BASE = "https://commons.wikimedia.org"

CUCUMBER_CATEGORIES = [
    "Cucumbers",
    "Cucumis sativus",
    "Cucumber dishes",
    "Pickling cucumbers",
    "Cucumber slices",
]

MIXER_CATEGORIES = [
    "Concrete mixers",
    "Concrete mixer trucks",
    "Rigid cement mixer trucks",
    "Cement mixer trucks (front discharge)",
    "Liebherr cement mixers",
    "MAN concrete mixer trucks",
    "Mercedes-Benz concrete mixer trucks",
    "Schwing Stetter (Memmingen)",
    "Concrete mixer semi-trailer trucks",
    "CIFA cement mixers",
    "Carmix concrete mixers",
]


class CommonsGalleryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.image_urls: set[str] = set()
        self.next_pages: set[str] = set()

    @staticmethod
    def _normalize_image_url(value: str) -> str | None:
        if not value:
            return None
        value = value.strip().split(" ", 1)[0]
        if value.startswith("//"):
            value = "https:" + value
        elif value.startswith("/"):
            value = urllib.parse.urljoin(COMMONS_BASE, value)
        if "upload.wikimedia.org" not in value or "/thumb/" not in value:
            return None
        lower = value.lower().split("?", 1)[0]
        if not lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
            return None
        # Gallery thumbnails are commonly 120-240 px. Ask the Wikimedia CDN for
        # a larger derivative while keeping the same original file.
        value = re.sub(r"/\d+px-([^/]+)$", r"/1024px-\1", value)
        return value

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: value for key, value in attrs if value is not None}
        if tag == "img":
            srcset = data.get("srcset", "")
            options = [part.strip() for part in srcset.split(",") if part.strip()]
            if options:
                for option in reversed(options):
                    normalized = self._normalize_image_url(option)
                    if normalized:
                        self.image_urls.add(normalized)
                        return
            normalized = self._normalize_image_url(data.get("src", ""))
            if normalized:
                self.image_urls.add(normalized)
            return

        if tag == "a":
            href = data.get("href", "")
            title = data.get("title", "")
            if "pagefrom=" in href and "/wiki/Category:" in href:
                self.next_pages.add(urllib.parse.urljoin(COMMONS_BASE, href))
            elif title.lower().startswith("next page") and href:
                self.next_pages.add(urllib.parse.urljoin(COMMONS_BASE, href))


def request_html(url: str, retries: int = 5) -> str:
    headers = {
        "User-Agent": builder.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=45)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(12, 2 ** attempt))
                continue
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def category_url(category: str) -> str:
    title = "Category:" + category.replace(" ", "_")
    return f"{COMMONS_BASE}/wiki/{urllib.parse.quote(title, safe=':_()-')}"


def scrape_category_images(category: str, max_pages: int = 5) -> list[str]:
    queue = [category_url(category)]
    visited: set[str] = set()
    images: set[str] = set()

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            html = request_html(url)
        except Exception as exc:
            print(f"[commons-cdn] {category}: {exc}", flush=True)
            continue
        parser = CommonsGalleryParser()
        parser.feed(html)
        images.update(parser.image_urls)
        for next_url in sorted(parser.next_pages):
            if next_url not in visited and next_url not in queue:
                queue.append(next_url)
        print(
            f"[commons-cdn] {category}: pages={len(visited)}, images={len(images)}",
            flush=True,
        )

    return sorted(images)


def category_candidates(
    class_name: str,
    categories: list[str],
) -> list[base.Candidate]:
    candidates: list[base.Candidate] = []
    for category in categories:
        landing = category_url(category)
        for image_url in scrape_category_images(category):
            candidates.append(
                base.Candidate(
                    class_name=class_name,
                    source="wikimedia_commons_cdn",
                    query=f"Category:{category}",
                    image_url=image_url,
                    landing_url=landing,
                    title=urllib.parse.unquote(Path(image_url).name),
                    creator="See Wikimedia Commons category/file page",
                    license="Wikimedia Commons; see source page",
                    license_url=landing,
                    source_id=image_url,
                )
            )
    unique = base.unique_candidates(candidates)
    random.Random(builder.SEED + len(class_name) * 911).shuffle(unique)
    print(
        f"[commons-cdn] {class_name}: raw={len(candidates)}, unique={len(unique)}",
        flush=True,
    )
    return unique


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name == "cucumber":
        categories = CUCUMBER_CATEGORIES
    elif class_name == "concrete_mixer":
        categories = MIXER_CATEGORIES
    else:
        raise ValueError(f"Unsupported class: {class_name}")
    candidates = category_candidates(class_name, categories)
    if len(candidates) < 220:
        raise RuntimeError(
            f"Direct Commons gallery source found only {len(candidates)} {class_name} images"
        )
    return candidates


builder.collect_target_candidates = collect_target_candidates


def direct_download_candidates(
    candidates: list[base.Candidate],
    download_root: Path,
    max_downloads: int,
    workers: int,
) -> list[base.Candidate]:
    if not candidates:
        return []
    class_name = candidates[0].class_name
    selected = candidates[: min(max_downloads, 700)]
    target_valid = 260
    result: list[base.Candidate] = []
    batch_size = 64

    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=5) as executor:
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
            f"[direct-download] {class_name}: "
            f"attempted={min(start + len(chunk), len(selected))}/{len(selected)}, "
            f"valid={len(result)}",
            flush=True,
        )
        if len(result) >= target_valid:
            break

    result.sort(key=base.stable_key)
    return result


base.download_candidates = direct_download_candidates


def main() -> int:
    return v7.v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
