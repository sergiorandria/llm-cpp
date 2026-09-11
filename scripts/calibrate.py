#!/usr/bin/env python3
"""F58: calibration for quantization — byte-frequency salient scales.

wte rows correspond to byte tokens, so token frequency is the activation
magnitude proxy (AWQ-style): salient rows = frequent bytes.
Writes scales.json {byte_id: scale} with scale=(freq/max)^alpha.

Usage:
  python3 scripts/calibrate.py --in data/calib/calib.txt --out scales.json
  python3 scripts/calibrate.py --check  # self-test: scales non-uniform
"""
import argparse, json, sys
from collections import Counter

def calibrate(path, alpha=0.5):
    with open(path, "rb") as f:
        raw = f.read()
    freq = Counter(raw)
    mx = max(freq.values())
    return {str(b): round((c / mx) ** alpha, 6) for b, c in sorted(freq.items())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/calib/calib.txt")
    ap.add_argument("--out", default="scales.json")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    scales = calibrate(a.inp, a.alpha)
    if a.check:
        vals = list(scales.values())
        assert len(vals) > 10, "too few byte values"
        assert min(vals) < max(vals), "scales must be non-uniform"
        assert all(0 < v <= 1.0 for v in vals)
        print(f"[calib] check ok: {len(vals)} bytes, range [{min(vals)}, {max(vals)}]")
        return 0
    with open(a.out, "w") as f:
        json.dump(scales, f)
    print(f"[calib] wrote {a.out} ({len(scales)} byte scales)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
