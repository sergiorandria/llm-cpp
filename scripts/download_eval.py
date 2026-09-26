#!/usr/bin/env python3
"""Full-set MMLU / HellaSwag downloader (stdlib only, mirrors download_data.py).

MMLU upstream (cais/mmlu) ships per-subject CSV; rows are converted to our
JSONL shape {"question","choices"[4],"answer" int,"subject"} which
load_mmlu_jsonl reads. HellaSwag validation split is already JSONL in the
Rowan/hellaswag shape (ctx/endings/label, occasional ctx_a/ctx_b).

Fair-use note: upstream sets are research benchmarks with their own
licenses (MMLU: MIT, HellaSwag: MIT). Downloaded files are NOT committed;
data/eval/ tracks only the tiny handcrafted samples + README.

Usage:
  python3 scripts/download_eval.py --dry-run
  python3 scripts/download_eval.py --dataset mmlu --subjects elementary_mathematics,astronomy
  python3 scripts/download_eval.py --dataset hellaswag
"""
import argparse, json, os, sys, time, urllib.request

# MMLU upstream is parquet (no stdlib reader), so page the HuggingFace
# datasets-server JSON API instead: rows carry question/subject/choices
# (list[str])/answer (int 0-3) — the exact shapes load_mmlu_jsonl reads.
DS_SERVER = "https://datasets-server.huggingface.co/rows"
DEFAULT_SUBJECTS = ["elementary_mathematics", "astronomy", "high_school_biology"]


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch_page(url, retries=4):
    """GET JSON with backoff; raises on persistent failure (no silent partial)."""
    err = None
    for attempt in range(retries):
        try:
            return json.loads(fetch(url).decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - retry then raise
            err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {err}")


def mmlu_subject(subject, out_fh):
    """Page one MMLU subject test split, write converted JSONL rows."""
    n, offset = 0, 0
    while True:
        url = (f"{DS_SERVER}?dataset=cais/mmlu&config={subject}"
               f"&split=test&offset={offset}&length=100")
        page = fetch_page(url)
        rows = page.get("rows", [])
        if not rows:
            break
        for entry in rows:
            row = entry.get("row", {})
            q, ch, ans = row.get("question"), row.get("choices"), row.get("answer")
            if (not q or not isinstance(ch, list) or len(ch) < 2 or
                    not isinstance(ans, int) or ans < 0 or ans >= len(ch)):
                continue
            out_fh.write(json.dumps({"question": q, "choices": ch, "answer": ans,
                                     "subject": row.get("subject", subject)}) + "\n")
            n += 1
        offset += len(rows)
        if offset >= page.get("num_rows_total", offset):
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmlu", choices=["mmlu", "hellaswag"])
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS),
                    help="comma-separated MMLU subjects (test split)")
    ap.add_argument("--out", default=None, help="output JSONL (default data/eval/full/<dataset>.jsonl)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dataset == "mmlu":
        subjects = [s.strip() for s in a.subjects.split(",") if s.strip()]
        out = a.out or "data/eval/full/mmlu.jsonl"
        print(f"[download] mmlu subjects={subjects} -> {out}")
        if a.dry_run:
            print("[download] dry-run ok (no network)")
            return 0
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        total = 0
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for s in subjects:
                    try:
                        n = mmlu_subject(s, fh)
                    except RuntimeError as e:
                        print(f"[download] subject {s} failed: {e}", file=sys.stderr)
                        raise
                    print(f"[download] {s}: {n} rows")
                    total += n
        except RuntimeError:
            if os.path.exists(tmp):
                os.remove(tmp)
            print("[download] INCOMPLETE — no file written (retry later)", file=sys.stderr)
            return 1
        if total == 0:
            os.remove(tmp)
            print("[download] nothing fetched", file=sys.stderr)
            return 1
        os.replace(tmp, out)
        print(f"[download] saved {out} ({total} rows)")
        return 0

    out = a.out or "data/eval/full/hellaswag_val.jsonl"
    print(f"[download] hellaswag validation -> {out}")
    if a.dry_run:
        print("[download] dry-run ok (no network)")
        return 0
    # Rowan/hellaswag via datasets-server (upstream GitHub raw moved).
    # NOTE: HF serves label as a digit-STRING ("3"); normalize to int here so
    # the file matches load_hellaswag_jsonl's primary shape (it also accepts
    # digit-strings, but normalized files are cleaner).
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    ok, offset, expected = 0, 0, None
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            while True:
                url = (f"{DS_SERVER}?dataset=Rowan/hellaswag&config=default"
                       f"&split=validation&offset={offset}&length=100")
                page = fetch_page(url)
                rows = page.get("rows", [])
                if expected is None:
                    expected = page.get("num_rows_total", 0)
                if not rows:
                    break
                for entry in rows:
                    o = entry.get("row", {})
                    endings = o.get("endings")
                    ctx = o.get("ctx") or " ".join(
                        s for s in (o.get("ctx_a", ""), o.get("ctx_b", "")) if s)
                    try:
                        lab = int(str(o.get("label")))
                    except (TypeError, ValueError):
                        continue
                    if (not ctx or not isinstance(endings, list) or len(endings) < 2 or
                            lab < 0 or lab >= len(endings)):
                        continue
                    fh.write(json.dumps({"ctx": ctx, "endings": endings, "label": lab,
                                         "activity_label": o.get("activity_label", "")}) + "\n")
                    ok += 1
                offset += len(rows)
                if offset >= page.get("num_rows_total", offset):
                    break
    except RuntimeError as e:
        print(f"[download] {e}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return 1
    print(f"[download] {ok} valid rows (server total {expected})")
    if ok == 0 or (expected and ok < expected):
        if os.path.exists(tmp):
            os.remove(tmp)
        print("[download] INCOMPLETE — no file written (retry later)", file=sys.stderr)
        return 1
    os.replace(tmp, out)
    print(f"[download] saved {out} ({os.path.getsize(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
