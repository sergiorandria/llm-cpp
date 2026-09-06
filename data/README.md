# Data

Place `input.txt` for training (e.g., TinyStories).

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/input.txt
```

## Current Corpus (2026-09-06) — Expanded

- **Malagasy raw**: 1768 docs, 2.38M chars, 364k words
  - tononkira: 487 lyrics (530k chars) — scraped serasera.org tononkira 10+ pages
  - vetso: 486 poems (334k chars) — serasera.org vetso
  - ohabolana: 600 proverbs (34k) — ohabolana.org
  - takila: 176 stories (1.46M) — takila.serasera.org
  - wikipedia: 19 articles (21k) — mg.wikipedia.org API
- **Malagasy augmented (EDA)**: 27k docs, 4.1M chars — word shuffle (15%) + deletion/duplication (8%/3%) via `data/malagasy/processed/augmented_mg.txt`
- **Shakespeare (English)**: 1.1M chars (202k words) — TinyShakespeare
- **Combined `data/input.txt`**: **7.76M chars, 1.22M words, ~59k docs, 101k lines** — Malagasy 2.38M + Shakespeare 1.1M + Augmented 4.1M

Stats: `data/malagasy/stats.json` + `data/malagasy/STATS.md`

To regenerate:
```bash
python scripts/scrape_malagasy.py --pages 10 --delay 1.0
python -c "import sys; sys.path.insert(0,'scripts'); from scrape_malagasy import build_processed; build_processed()"
cat data/malagasy/processed/shakespeare.txt >> data/input.txt
python scripts/augment_mg.py  # EDA augmentation
```

For training:
```bash
./build/llm-cpp train --config config/config.json --data data/input.txt
# quick test (tiny model):
./build/llm-cpp train --config config/tiny.json --data data/input.txt --max-iters 10
```
