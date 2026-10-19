"""Download Goodfire's published layer-0 down_proj programs/windows (p-82af68dd) into data/.

Usage: python fetch.py [component ...]   # shards for the given components (default: all 96)
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://goodfire-ai.github.io/param-decomp/data"
SHARD = 64
OUT = Path(__file__).parent / "data"


def get(rel):
    path = OUT / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{rel}", path)
    return path


if __name__ == "__main__":
    for rel in ["summary.json", "vocab.json", "components/index.json"]:
        get(rel)
    comps = [int(c) for c in sys.argv[1:]] or range(0, 6144, SHARD)
    for s in sorted({c // SHARD for c in comps}):
        print(get(f"components/s{s:03d}.json"))
