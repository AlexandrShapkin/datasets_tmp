#!/usr/bin/env python3
from __future__ import annotations

import urllib.parse

import build_cucumber_mixer_open_set_v9 as v9


def existing_thumbnail_url(value: str) -> str | None:
    if not value:
        return None
    value = value.strip().split(" ", 1)[0]
    if value.startswith("//"):
        value = "https:" + value
    elif value.startswith("/"):
        value = urllib.parse.urljoin(v9.COMMONS_BASE, value)

    lower = value.lower().split("?", 1)[0]
    if "upload.wikimedia.org" not in value or "/thumb/" not in value:
        return None
    if not lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
        return None

    # Keep the largest URL already published in the category page's srcset.
    # Unlike a constructed 1024-pixel derivative, this URL is guaranteed to
    # exist even when the original image is small.
    return value


v9.CommonsGalleryParser._normalize_image_url = staticmethod(existing_thumbnail_url)


def main() -> int:
    return v9.main()


if __name__ == "__main__":
    raise SystemExit(main())
