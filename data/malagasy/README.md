# Malagasy Corpus — Data Collection

Scraped Malagasy language data from public sources for LLM training with `llm-cpp`.

**Created:** 2026-09-06  
**Language:** Malagasy (`mg`) — plus occasional French/Code-switching in lyrics  
**License:** Scraped content retains original site rights; provided for **research/education** (fair use). Respect `serasera.org` ToS; do not redistribute commercially without permission.

## Sources

| Source | Site | Description | Raw | Processed | Pages scraped |
|--------|------|-------------|-----|-----------|---------------|
| **Tononkira** | `tononkira.serasera.org` | Song lyrics (tononkira) — 17k total available | `raw/tononkira/tononkira.jsonl` | `processed/tononkira.txt` | 20 (401 links, 487 lyrics fetched) |
| **Vetso** | `vetso.serasera.org` | Poems (tononkalo) — 19k total | `raw/vetso/vetso.jsonl` | `processed/vetso.txt` | 20 (401 links, 486 poems) |
| **Ohabolana** | `ohabolana.org` | Proverbs (ohabolana) — 6.5k total | `raw/ohabolana/ohabolana.jsonl` | `processed/ohabolana.txt` | 20 (600 proverbs) |
| **Takila** | `takila.serasera.org` | Stories / tales / publications (tantara, angano, kabary, literatiora kristiana) | `raw/takila/takila.jsonl` | `processed/takila.txt` | 20 (176 stories) |
| **Wikipedia** | `mg.wikipedia.org` | Encyclopedia extracts via API | `raw/wikipedia/mg.wikipedia.jsonl` | `processed/wikipedia.txt` | 11 articles (API, delay-respecting) |

> **Total (2026-09-06):** `1760` documents, `2,374,297` chars (~2.4M), `362,938` words.  
> Breakdown in `stats.json` and `STATS.md`.

## Folder Structure

```
data/
  input.txt                      # combined corpus for llm-cpp (symlink of malagasy/processed/input.txt)
  malagasy/
    README.md                    # this file
    stats.json                   # machine-readable stats
    STATS.md                     # human readable
    raw/
      tononkira/tononkira.jsonl      # {url, title, artist, lyrics}
      tononkira/tononkira_links.json
      tononkira/html_samples/        # 3 sample HTMLs for debugging
      vetso/vetso.jsonl              # {url, title, author, tags, poem}
      ohabolana/ohabolana.jsonl      # {url, proverb, categories}
      takila/takila.jsonl            # {url, title, author, content}
      wikipedia/mg.wikipedia.jsonl   # {title, extract, url}
    processed/
      tononkira.txt   # title - artist \n lyrics \n\n
      vetso.txt       # title - author \n poem \n\n
      ohabolana.txt   # one proverb per line
      takila.txt      # title \n content \n\n
      wikipedia.txt   # title \n extract \n\n
      input.txt       # shuffled combined (also copied to data/input.txt)
```

## Usage

### Training with llm-cpp

```bash
# input.txt already at data/input.txt (Karpathy style)
ls -lh data/input.txt data/malagasy/processed/input.txt
wc -m data/input.txt  # 2.37M chars

# Train (example, adjust config)
./build/llm-cpp train --config config/config.json --data data/input.txt
# Or Python:
python scripts/train.py --data data/input.txt --out checkpoints/malagasy
```

### Regenerate / Extend

```bash
# 5 pages quick test (~300 docs)
python scripts/scrape_malagasy.py --pages 5 --delay 0.8 --wiki-articles 100

# 20 pages (~1.7k docs, current)
python scripts/scrape_malagasy.py --pages 20 --delay 0.6 --wiki-articles 200 --sources tononkira vetso ohabolana takila

# Full (all pages, ~hours, be nice: delay >=1.0)
python scripts/scrape_malagasy.py --full --delay 1.2 --wiki-articles 500

# Only wikipedia with backoff
python scripts/scrape_malagasy.py --sources vetso --pages 20 --delay 0.8 --skip-wiki
python scripts/scrape_malagasy.py --pages 0 --wiki-articles 1000 --sources takila --skip-wiki  # then manually run wiki part via python -c
```

Script features:
- **Resume**: skips already fetched URLs (checks `*.jsonl`)
- **Rate limit**: `delay + random(0,0.5)` between requests, `User-Agent` identification, 429 back-off
- **Deduplication**: per-list + global link dedup
- **Robust parsing**: `BeautifulSoup+lxml`, fallback for malformed HTML, unicode `utf-8`

### Verification

```bash
cat data/malagasy/stats.json | python -m json.tool
wc -l data/malagasy/raw/*/*.jsonl
wc -w data/malagasy/processed/*.txt data/input.txt
head -n 50 data/malagasy/processed/input.txt
head -n 30 data/malagasy/processed/tononkira.txt
head -n 20 data/malagasy/processed/ohabolana.txt
```

## Example Samples

**Tononkira (lyrics)**
```
Ho Paradisa - Andoniaina
Efa miaharatsy ny fiainana an-tany
Feno fijaliana feno tomany
...
Ho paradisa ny tany
Indray andro any
```

**Vetso (poem)**
```
Fijeriko anao - Herilanja RANDRIAMANGA
Fijeriko anao lasan'olon'iny ...
Ny ao anatiko ao, vaky toa siny...
```

**Ohabolana (proverb)**
```
Ny fahoriana tsy misy mpila, fa ny zarany ihany no zakaina.
Aza mifampidera taim-by fa samy zanaky ny mpanefy
```

**Takila (story)**
```
Ny "zava-miafin'ny toe-panahy ..." - Elysé Azrael
Amin'izao fotoanan'ny liberalisme sy ny post-modernisme izao, dia "samy manao izay mahafalifaly ny fony" ...
```

## Ethics & Legal

- All sources are **publicly accessible** without paywall. Scraper respects `robots.txt` implicitly via modest rate (0.6–1.2s delay) and non-aggressive concurrent=1.
- **Do not hammer**: default 20 pages ≈ 400 items per source; full scrape is ~870 pages for tononkira → use `--full --delay 1.5` only if needed.
- **Attribution**: keep `url` field in `raw/*.jsonl` for provenance. If publishing corpus, credit `serasera.org` community authors.
- **Removal**: if you are rights-holder and want removal, delete corresponding lines from `processed/input.txt` and raw jsonl.

## Future Extensions

- `mg.wikisource.org` (books), `rakibolana.org` (dictionary), `dumps.wikimedia.org/mgwiki` (full dump 387M), `huggingface.co/datasets` Malagasy corpora (OSCAR, CC100).
- Add `hainteny` (traditional poetry) via `gutenberg.org` Hainteny book (public domain) and `aandc` corpus.
- Increase takila pages → currently 176; site likely has >1000 stories (increase `--pages 100`).
- Wikipedia: fix 429 handling (exponential backoff) to fetch 500+ articles; or download dump: `wget https://dumps.wikimedia.org/mgwiki/latest/mgwiki-latest-pages-articles.xml.bz2`.

## Script Location

`scripts/scrape_malagasy.py:1` — entry point; see `--help` for options.

```bash
python scripts/scrape_malagasy.py --help
```
