#!/usr/bin/env python
"""Download the HPO ontology file (hp.obo) to data/hpo/."""
from __future__ import annotations

import sys
from pathlib import Path

import requests

HP_OBO_URL = "https://purl.obolibrary.org/obo/hp.obo"
OUT_PATH = Path("data/hpo/hp.obo")


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HP_OBO_URL} -> {OUT_PATH}")
    r = requests.get(HP_OBO_URL, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    with open(OUT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"  {downloaded/1024:.0f} KB / {total/1024:.0f} KB ({pct:.1f}%)", end="\r")
    print(f"\nSaved to {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
