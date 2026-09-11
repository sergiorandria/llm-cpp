#!/usr/bin/env python3
"""G68: model card generator — arch + params + quant + usage docs.

Usage:
  python3 scripts/model_card.py --config config/config.json.example [--checkpoint checkpoints/model.bin] [--out MODEL_CARD.md]
"""
import argparse, json, os, struct, sys

def read_ckpt(path):
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        if magic != 0x4C4C4D00:
            return {"ok": False, "error": f"bad magic 0x{magic:08x}"}
        sizes = struct.unpack("<5Q", f.read(40))
        return {"ok": True, "version": ver, "vocab": sizes[0], "layers": sizes[1],
                "heads": sizes[2], "embd": sizes[3], "block": sizes[4],
                "bytes": os.path.getsize(path)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json.example")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="MODEL_CARD.md")
    a = ap.parse_args()
    cfg = {}
    try:
        with open(a.config) as f:
            txt = f.read()
        import re
        for k in ["vocab_size", "n_layers", "n_heads", "n_embd", "block_size"]:
            m = re.search(r'"%s"\s*:\s*(\d+)' % k, txt)
            if m:
                cfg[k] = int(m.group(1))
    except FileNotFoundError:
        pass
    ckpt = read_ckpt(a.checkpoint) if a.checkpoint else {"ok": False, "error": "no checkpoint given"}
    n_params = None
    if cfg:
        v, l, e = cfg.get("vocab_size", 0), cfg.get("n_layers", 0), cfg.get("n_embd", 0)
        n_params = v * e * 2 + l * (12 * e * e) if v and l and e else None
    lines = ["# Model Card (llm-cpp)", ""]
    lines.append("## Architecture")
    for k in ["vocab_size", "n_layers", "n_heads", "n_embd", "block_size"]:
        lines.append(f"- {k}: {cfg.get(k, '?')}")
    if n_params:
        lines.append(f"- params (est.): ~{n_params:,}")
    lines += ["", "## Checkpoint"]
    if ckpt.get("ok"):
        lines.append(f"- file: {a.checkpoint} ({ckpt['bytes']} bytes, binary v{ckpt['version']})")
        lines.append(f"- vocab={ckpt['vocab']} layers={ckpt['layers']} heads={ckpt['heads']} embd={ckpt['embd']} block={ckpt['block']}")
    else:
        lines.append(f"- {ckpt.get('error', '?')}")
    lines += ["", "## Usage",
              "```bash",
              "./build/llm-cpp generate --prompt \"Hello\" --max_tokens 50",
              "./build/llm-cpp serve --port 8080",
              "curl -X POST localhost:8080/v1/completions -d '{\"prompt\":\"Hi\",\"max_tokens\":20}'",
              "```", "", "## License", "MIT — see LICENSE.", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[card] wrote {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
