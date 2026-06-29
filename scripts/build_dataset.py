#!/usr/bin/env python3
from __future__ import annotations

import sys

import build_dataset_v2 as builder


def enforce_minimum(flag: str, minimum: int) -> None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        sys.argv.extend([flag, str(minimum)])
        return
    if index + 1 >= len(sys.argv):
        sys.argv.append(str(minimum))
        return
    try:
        current = int(sys.argv[index + 1])
    except ValueError:
        current = 0
    sys.argv[index + 1] = str(max(current, minimum))


enforce_minimum("--candidate-limit", 1200)
enforce_minimum("--source-limit", 250)
enforce_minimum("--workers", 20)


if __name__ == "__main__":
    raise SystemExit(builder.base.main())
