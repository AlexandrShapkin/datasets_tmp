#!/usr/bin/env python3
from __future__ import annotations

import random
import time
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

import build_cucumber_mixer_open_set_v6 as v6

builder = v6.builder
base = builder.base

COMMONS_BASE = "https://commons.wikimedia.org"

CUCUMBER_CATEGORIES = [
    "Cucumbers",
    "Cucumis sativus",
    "Sliced cucumbers",
    "Cucumbers in markets",
    "Cucumbers on plants",
    "Cucumbers by country",
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

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


class FileLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.file_titles: set[str] = set()
        self.next_pages: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        data = {key: value for key, value in attrs if value is not None}
        href = data.get("href", "")
        title = data.get("title", "")

        if href.startswith("/wiki/File:"):
            encoded = href.split("/wiki/File:", 1)[1].split("#", 1)[0]
            filename = urllib.parse.unquote(encoded).replace("_", " ").strip()
            if Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
                self.file_titles.add(filename)

        # Category pages with more than 200 items expose a continuation link.
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
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Failed to fetch category page {url}: {last_error}")


def category_url(category: str) -> str:
    title = "Category:" + category.replace(" ", "_")
    return f"{COMMONS_BASE}/wiki/{urllib.parse.quote(title, safe=':_()-')}"


def file_page_url(filename: str) -> str:
    title = "File:" + filename.replace(" ", "_")
    return f"{COMMONS_BASE}/wiki/{urllib.parse.quote(title, safe=':_()-')}"


def file_image_url(filename: str) -> str:
    encoded = urllib.parse.quote(filename.replace(" ", "_"), safe="")
    return f"{COMMONS_BASE}/wiki/Special:Redirect/file/{encoded}?width=1200"


def scrape_category_files(category: str, max_pages: int = 5) -> list[str]:
    queue = [category_url(category)]
    visited: set[str] = set()
    files: set[str] = set()

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            html = request_html(url)
        except Exception as exc:
            print(f"[commons-html] {category}: failed {url}: {exc}", flush=True)
            continue
        parser = FileLinkParser()
        parser.feed(html)
        files.update(parser.file_titles)
        for next_url in sorted(parser.next_pages):
            if next_url not in visited and next_url not in queue:
                queue.append(next_url)
        print(
            f"[commons-html] {category}: pages={len(visited)}, files={len(files)}",
            flush=True,
        )

    return sorted(files)


def commons_html_candidates(
    class_name: str,
    categories: list[str],
) -> list[base.Candidate]:
    candidates: list[base.Candidate] = []
    for category in categories:
        filenames = scrape_category_files(category)
        for filename in filenames:
            candidates.append(
                base.Candidate(
                    class_name=class_name,
                    source="wikimedia_commons_html",
                    query=f"Category:{category}",
                    image_url=file_image_url(filename),
                    landing_url=file_page_url(filename),
                    title=filename,
                    creator="See Wikimedia Commons file page",
                    license="Wikimedia Commons; see file page",
                    license_url=file_page_url(filename),
                    source_id=filename,
                )
            )
    unique = base.unique_candidates(candidates)
    random.Random(builder.SEED + len(class_name) * 313).shuffle(unique)
    print(
        f"[commons-html] {class_name}: raw={len(candidates)}, unique={len(unique)}",
        flush=True,
    )
    return unique


def collect_target_candidates(class_name: str) -> list[base.Candidate]:
    if class_name == "cucumber":
        kaggle = v6.v5.v4.v3.kaggle_cucumber_candidates()
        commons = commons_html_candidates(class_name, CUCUMBER_CATEGORIES)
        combined = base.unique_candidates(kaggle + commons)
        random.Random(builder.SEED + 701).shuffle(combined)
        print(
            f"[source] cucumber: kaggle={len(kaggle)}, commons={len(commons)}, "
            f"combined={len(combined)}",
            flush=True,
        )
        return combined

    if class_name == "concrete_mixer":
        commons = commons_html_candidates(class_name, MIXER_CATEGORIES)
        if len(commons) < 150:
            raise RuntimeError(
                f"Static Commons scraper found only {len(commons)} mixer files"
            )
        return commons

    raise ValueError(f"Unsupported class: {class_name}")


builder.collect_target_candidates = collect_target_candidates


def main() -> int:
    return v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
