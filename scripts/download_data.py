#!/usr/bin/env python3
"""B19: dataset downloader with SHA256 verify (TinyStories/Shakespeare).

Usage:
  python3 scripts/download_data.py --dry-run
  python3 scripts/download_data.py --dataset tinystories --out data/input.txt
"""
import argparse, hashlib, os, sys, urllib.request

SOURCES = {
    "tinystories": ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt", None),
    "shakespeare": ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                    "1c571c4daf0d34b17ad3d9b7442c2ac4e5e6a15ab9a28d9978d62a65946d8bdc"),
}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shakespeare", choices=list(SOURCES))
    ap.add_argument("--out", default="data/input.txt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    url, sha = SOURCES[a.dataset]
    print(f"[download] {a.dataset} <- {url}")
    if a.dry_run:
        print("[download] dry-run ok (no network)")
        return 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    if sha:
        got = sha256_file(tmp)
        if got != sha:
            print(f"[download] CHECKSUM MISMATCH got={got} want={sha}", file=sys.stderr)
            os.remove(tmp)
            return 1
        print(f"[download] sha256 ok {got[:16]}…")
    os.replace(tmp, a.out)
    print(f"[download] saved {a.out} ({os.path.getsize(a.out)} bytes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
