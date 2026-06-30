#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

queries = [
    "concrete mixer",
    "cement mixer",
    "construction vehicles",
    "construction equipment images",
    "cucumber images",
    "fruit vegetable image recognition",
    "vegetable classification cucumber",
]

lines: list[str] = []
for query in queries:
    lines.append(f"\n===== QUERY: {query} =====")
    process = subprocess.run(
        ["kaggle", "datasets", "list", "-s", query, "--csv"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    lines.append(f"returncode={process.returncode}")
    lines.append(process.stdout[:50000])
    if process.stderr:
        lines.append("STDERR:")
        lines.append(process.stderr[:10000])

output = Path("dist_kaggle_search/kaggle_search.txt")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(lines), encoding="utf-8")
print(output.read_text(encoding="utf-8"))
