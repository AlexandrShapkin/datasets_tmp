#!/usr/bin/env python3
from __future__ import annotations

import json
import traceback
from pathlib import Path

import build_cucumber_mixer_open_set_v11 as builder_v11

builder = builder_v11

if __name__ == "__main__":
    try:
        raise SystemExit(builder.main())
    except SystemExit:
        raise
    except BaseException as error:
        output = Path("dist_open_set")
        output.mkdir(parents=True, exist_ok=True)
        trace = traceback.format_exc()
        (output / "build_error.txt").write_text(trace, encoding="utf-8")
        (output / "build_error.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": trace,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(trace, flush=True)
        raise SystemExit(1)
