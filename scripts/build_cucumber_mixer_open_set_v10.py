#!/usr/bin/env python3
from __future__ import annotations

import urllib.parse

import build_cucumber_mixer_open_set_v9 as v9


def original_commons_url(value: str) -> str | None:
    if not value:
        return None
    value = value.strip().split(" ", 1)[0]
    if value.startswith("//"):
        value = "https:" + value
    elif value.startswith("/"):
        value = urllib.parse.urljoin(v9.COMMONS_BASE, value)

    marker = "/wikipedia/commons/thumb/"
    if "upload.wikimedia.org" not in value or marker not in value:
        return None

    prefix, remainder = value.split(marker, 1)
    parts = remainder.split("/")
    if len(parts) < 4:
        return None

    # Thumbnail URL format:
    # .../commons/thumb/<hash1>/<hash2>/<original-name>/<width>-<original-name>
    # Original URL format:
    # .../commons/<hash1>/<hash2>/<original-name>
    original_path = "/".join(parts[:-1])
    original_url = prefix + "/wikipedia/commons/" + original_path
    lower = original_url.lower().split("?", 1)[0]
    if not lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
        return None
    return original_url


v9.CommonsGalleryParser._normalize_image_url = staticmethod(original_commons_url)


def main() -> int:
    return v9.main()


if __name__ == "__main__":
    raise SystemExit(main())
