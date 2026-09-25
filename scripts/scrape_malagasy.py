#!/usr/bin/env python3
"""
Scrape Malagasy data from multiple public sources for LLM training.
Sources:
  - tononkira.serasera.org (lyrics) ~ 17k songs
  - vetso.serasera.org (poems / tononkalo) ~ 19k poems
  - ohabolana.org (proverbs) ~ 6k ohabolana
  - takila.serasera.org (stories / publications) ~ many tales
  - mg.wikipedia.org (encyclopedia - via API)
  - Optional: Gutenberg / Wikisource mg

Output:
  data/malagasy/raw/<source>/
  data/malagasy/processed/<source>.txt
  data/malagasy/processed/input.txt  (combined for llm-cpp)
  data/malagasy/stats.json

Usage:
  python scripts/scrape_malagasy.py --pages 5  # quick test, 5 list pages per source
  python scripts/scrape_malagasy.py --pages 20 --delay 0.8
  python scripts/scrape_malagasy.py --full   # try to scrape all (respect rate limit)

Author: generated for llm-cpp Malagasy corpus project
"""

import os
import re
import json
import time
import argparse
import random
from pathlib import Path
from urllib.parse import urljoin, quote
import requests
from bs4 import BeautifulSoup

# Config
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "malagasy"
RAW_DIR = BASE_DIR / "raw"
PROC_DIR = BASE_DIR / "processed"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (MalagasyCorpusResearch; educational+non-commercial; contact: llm-cpp project) AppleWebKit/537.36"
}

# Sources meta
SOURCES = {
    "tononkira": {
        "base": "https://tononkira.serasera.org",
        "list_url": "https://tononkira.serasera.org/tononkira?page={page}",
        "detail_pattern": "/hira/",
        "pages_total": 870,
        "per_page": 20,
    },
    "vetso": {
        "base": "https://vetso.serasera.org",
        "list_url": "https://vetso.serasera.org/vetso?page={page}",
        "detail_pattern": "/tononkalo/",
        "pages_total": 966,
        "per_page": 20,
    },
    "ohabolana": {
        "base": "https://ohabolana.org",
        "list_url": "https://ohabolana.org/ohabolana?page={page}",
        "detail_pattern": "/ohabolana/vakio/",
        "pages_total": 212,
        "per_page": 30,
    },
    "takila": {
        "base": "https://takila.serasera.org",
        "list_url": "https://takila.serasera.org/tantara?page={page}",
        "detail_pattern": "/mpanoratra/",  # trickier, we extract story links directly
        "pages_total": 100,  # unknown, estimate
        "per_page": 20,
    }
}

def ensure_dirs():
    for d in [RAW_DIR, PROC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        (RAW_DIR / src).mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "wikipedia").mkdir(parents=True, exist_ok=True)

def fetch(url, session, delay=1.0, retries=3, timeout=20):
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0, 0.5))
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding or 'utf-8'
                return resp.text
            elif resp.status_code in (429, 503):
                wait = 5 * (attempt+1)
                print(f"  ! {resp.status_code} rate limited, wait {wait}s for {url}")
                time.sleep(wait)
            else:
                print(f"  ! HTTP {resp.status_code} for {url}")
                time.sleep(1)
        except Exception as e:
            print(f"  ! fetch error {e} for {url} attempt {attempt+1}")
            time.sleep(2)
    return None

def extract_links_from_list(html, base, pattern):
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # normalize
        if href.startswith("/"):
            href = urljoin(base, href)
        if pattern in href:
            # filter out unwanted: exclude /ankafizo, /hevitra, #hevitra, /ahitsio, /tabs, /audio, /user, /?username
            if any(x in href for x in ["/ankafizo", "/hevitra", "/ahitsio", "/tabs/", "/audio", "username=", "/ampidiro", "/sary/"]):
                continue
            # ensure not duplicate query pagination
            links.append(href)
    # dedup preserve order
    seen=set()
    uniq=[]
    for l in links:
        # strip fragment and query for clean? but keep as is
        l = l.split("#")[0]
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq

def parse_tononkira_detail(html, url):
    soup = BeautifulSoup(html, "lxml")
    text_full = soup.get_text(separator="\n")
    # title
    title_tag = soup.find("h2")
    title = title_tag.get_text(strip=True) if title_tag else ""
    # extract lyrics between Nalaina ... and --------
    # strategy: locate marker
    marker = "Nalaina tao amin"
    idx = text_full.find(marker)
    if idx != -1:
        # find end marker --------
        end = text_full.find("--------", idx)
        if end == -1:
            end = text_full.find("Rohy:", idx)
        segment = text_full[idx:end] if end != -1 else text_full[idx: idx+5000]
        # remove first line marker itself
        lines = segment.split("\n")
        # filter out the marker line
        cleaned_lines = []
        for line in lines:
            if "Nalaina tao" in line:
                continue
            cleaned_lines.append(line)
        lyrics = "\n".join(cleaned_lines).strip()
        # cleanup extra empty lines excessive
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
        # Remove stray site mentions inside? keep but trim
        lyrics = lyrics.strip()
    else:
        # fallback: try to find container with that text via soup
        lyrics = ""
    # metadata
    # extract artist via link /mpihira/
    artist = ""
    artist_link = soup.find("a", href=lambda x: x and "/mpihira/" in x)
    if artist_link:
        artist = artist_link.get_text(strip=True)
    # views, date etc
    return {
        "url": url,
        "title": title,
        "artist": artist,
        "lyrics": lyrics,
        "raw_text_length": len(text_full),
    }

def parse_vetso_detail(html, url):
    soup = BeautifulSoup(html, "lxml")
    text_full = soup.get_text(separator="\n")
    title_tag = soup.find("h2")
    title = title_tag.get_text(strip=True) if title_tag else ""
    marker = "Nalaina tao amin"
    idx = text_full.find(marker)
    if idx != -1:
        end = text_full.find("--------", idx)
        if end == -1:
            end = text_full.find("Rohy:", idx)
        segment = text_full[idx:end] if end != -1 else text_full[idx: idx+8000]
        lines = segment.split("\n")
        cleaned = [l for l in lines if "Nalaina tao" not in l]
        poem = "\n".join(cleaned).strip()
        poem = re.sub(r"\n{3,}", "\n\n", poem)
    else:
        poem = ""
    # author
    author = ""
    author_link = soup.find("a", href=lambda x: x and "/mpanoratra/" in x)
    if author_link:
        author = author_link.get_text(strip=True)
    # category tags
    tags = [a.get_text(strip=True) for a in soup.find_all("a", href=lambda x: x and "tags=" in str(x))]
    return {
        "url": url,
        "title": title,
        "author": author,
        "tags": tags,
        "poem": poem,
    }

def parse_ohabolana_detail(html, url):
    soup = BeautifulSoup(html, "lxml")
    # title is og:title or h1 style
    # The proverb text is main title
    title_meta = soup.find("meta", property="og:title")
    if title_meta and title_meta.get("content"):
        proverb = title_meta["content"].strip()
    else:
        # fallback parse from H? maybe find largest text
        h = soup.find("h1")
        proverb = h.get_text(strip=True) if h else soup.title.string.strip() if soup.title else ""
        proverb = proverb.split("-")[0].strip()
    text_full = soup.get_text(separator="\n")
    # explanation may be in og:description or in page?
    desc_meta = soup.find("meta", property="og:description")
    desc = desc_meta["content"].strip() if desc_meta and desc_meta.get("content") else ""
    # categories: look for "Sokajy:" section
    cats = []
    # find links with tags
    for a in soup.find_all("a", href=True):
        if "tags=" in a["href"] or "/ohabolana?tags=" in a["href"]:
            cats.append(a.get_text(strip=True))
    # heuristic: proverb is already extracted
    return {
        "url": url,
        "proverb": proverb,
        "description": desc,
        "categories": cats,
        "full_text": proverb,
    }

def parse_takila_detail(html, url):
    soup = BeautifulSoup(html, "lxml")
    text_full = soup.get_text(separator="\n")
    title_tag = soup.find("h1") or soup.find("h2") or soup.title
    title = title_tag.get_text(strip=True) if title_tag else url
    # Look for main story content: after Mpanoratra block, often inside div with content
    # Heuristic: find largest paragraph block
    # Try to find div class containing story: search for all p
    # Simpler: extract between "Mpanoratra" header and "Hametraka hevitra"
    start_markers = ["Mpanoratra", "Nampiditra", "Nitsidika"]
    end_marker = "Hametraka hevitra"
    # Get cleaned text lines
    lines = [l.strip() for l in text_full.split("\n") if l.strip()]
    # Find start index after metadata
    start_idx = 0
    for i, l in enumerate(lines):
        if "Pejy" in l and l.strip() == "Pejy 1":
            start_idx = i+1
            break
    end_idx = len(lines)
    for i, l in enumerate(lines):
        if end_marker in l:
            end_idx = i
            break
    content = "\n".join(lines[start_idx:end_idx]).strip()
    # Filter out very short or nav junk
    # If too short, fallback to article tag content
    if len(content) < 100:
        article = soup.find("article")
        if article:
            content = article.get_text(separator="\n").strip()
        else:
            # take all text between title and footer
            content = text_full.strip()
            content = re.sub(r"\n{3,}", "\n\n", content)[:8000]
    # cleanup
    content = re.sub(r"\n{3,}", "\n\n", content)
    # Extract author
    author = ""
    author_link = soup.find("a", href=lambda x: x and "/mpanoratra/" in x)
    if author_link:
        author = author_link.get_text(strip=True)
    return {
        "url": url,
        "title": title,
        "author": author,
        "content": content,
    }

def scrape_source(source_name, pages, delay, session):
    meta = SOURCES[source_name]
    base = meta["base"]
    pattern = meta["detail_pattern"]
    list_url_tpl = meta["list_url"]

    print(f"\n=== Scraping {source_name} ({pages} pages, delay {delay}s) ===")
    all_links = []
    for p in range(1, pages+1):
        url = list_url_tpl.format(page=p)
        print(f"[{source_name}] list page {p}/{pages}: {url}")
        html = fetch(url, session, delay=delay)
        if not html:
            print(f"  -> failed page {p}")
            continue
        if source_name == "takila":
            # Custom extraction for takila: story links are https://takila.serasera.org/<author>/<slug>
            soup = BeautifulSoup(html, "lxml")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = urljoin(base, href)
                # only takila domain
                if not href.startswith("https://takila.serasera.org/"):
                    continue
                if any(x in href for x in ["/ankafizo", "#hevitra", "/ahitsio", "/tantara", "/mpanoratra", "/login", "/hampiditra", "/tag", "page="]):
                    continue
                path = href.replace("https://takila.serasera.org/", "").strip("/").split("?")[0].split("#")[0]
                parts = path.split("/")
                if len(parts) == 2 and parts[0] and parts[1]:
                    # avoid pagination / single segment
                    clean = href.split("#")[0].split("?")[0].rstrip("/")
                    links.append(clean)
            # dedup preserve order
            links = list(dict.fromkeys(links))
        else:
            links = extract_links_from_list(html, base, pattern)

        print(f"  -> found {len(links)} links")
        all_links.extend(links)

    # Deduplicate globally
    all_links = list(dict.fromkeys(all_links))
    print(f"[{source_name}] total unique links: {len(all_links)}")

    # Save links list
    links_file = RAW_DIR / source_name / f"{source_name}_links.json"
    with open(links_file, "w", encoding="utf-8") as f:
        json.dump(all_links, f, ensure_ascii=False, indent=2)

    # Now fetch detail pages
    records = []
    output_jsonl = RAW_DIR / source_name / f"{source_name}.jsonl"
    # resume support: skip already processed urls
    existing_urls = set()
    if output_jsonl.exists():
        try:
            with open(output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        existing_urls.add(rec.get("url"))
                    except: pass
            print(f"  resume: {len(existing_urls)} already fetched")
        except: pass

    for idx, link in enumerate(all_links, 1):
        if link in existing_urls:
            continue
        print(f"[{source_name}] {idx}/{len(all_links)} fetch {link}")
        html = fetch(link, session, delay=delay)
        if not html:
            print("  -> failed")
            continue
        try:
            if source_name == "tononkira":
                rec = parse_tononkira_detail(html, link)
                # only keep if lyrics non-empty
                if len(rec.get("lyrics","").strip()) < 20:
                    print(f"  -> empty lyrics skip")
                    continue
            elif source_name == "vetso":
                rec = parse_vetso_detail(html, link)
                if len(rec.get("poem","").strip()) < 20:
                    print(f"  -> empty poem skip")
                    continue
            elif source_name == "ohabolana":
                rec = parse_ohabolana_detail(html, link)
                if len(rec.get("proverb","").strip()) < 5:
                    continue
            elif source_name == "takila":
                rec = parse_takila_detail(html, link)
                if len(rec.get("content","").strip()) < 50:
                    print(f"  -> short content skip ({len(rec.get('content',''))})")
                    continue
            else:
                rec = {"url": link}
        except Exception as e:
            print(f"  -> parse error {e}")
            continue
        records.append(rec)
        # append to jsonl immediately
        with open(output_jsonl, "a", encoding="utf-8") as out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # optional small raw html save for first few
        if idx <= 3:
            html_dir = RAW_DIR / source_name / "html_samples"
            html_dir.mkdir(exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", link.replace("https://",""))[:100]
            with open(html_dir / f"{safe}.html", "w", encoding="utf-8") as hf:
                hf.write(html)

    print(f"[{source_name}] fetched {len(records)} new records; total now {len(all_links)} links processed")
    return records

def scrape_wikipedia(num_articles=500, delay=0.8):
    """
    Scrape mg.wikipedia.org via API (not dump) to get article extracts in Malagasy.
    Uses allpages + extracts.
    """
    print(f"\n=== Scraping mg.wikipedia ({num_articles} articles) ===")
    session = requests.Session()
    base_api = "https://mg.wikipedia.org/w/api.php"
    # Step 1: get list of pages via allpages
    all_titles = []
    apcontinue = None
    while len(all_titles) < num_articles:
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": 50,
            "apfilterredir": "nonredirects",
            "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        try:
            r = session.get(base_api, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("allpages", [])
            for p in pages:
                all_titles.append(p["title"])
                if len(all_titles) >= num_articles:
                    break
            if "continue" in data:
                apcontinue = data["continue"].get("apcontinue")
                time.sleep(delay)
            else:
                break
        except Exception as e:
            print(f" wiki list error {e}")
            break
    print(f" wiki: got {len(all_titles)} titles")
    # Step 2: fetch extracts in batches of 20
    extracts = []
    out_path = RAW_DIR / "wikipedia" / "mg.wikipedia.jsonl"
    # resume
    existing = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    existing.add(rec.get("title"))
                except: pass
    # batch
    for i in range(0, len(all_titles), 20):
        batch = [t for t in all_titles[i:i+20] if t not in existing]
        if not batch:
            continue
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "titles": "|".join(batch),
            "format": "json",
        }
        print(f" wiki batch {i//20+1} / {len(all_titles)//20+1}")
        try:
            r = session.get(base_api, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, info in pages.items():
                if "missing" in info:
                    continue
                title = info.get("title")
                extract = info.get("extract", "")
                if not extract or len(extract.strip()) < 50:
                    continue
                rec = {"title": title, "pageid": info.get("pageid"), "extract": extract, "url": f"https://mg.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"}
                extracts.append(rec)
                with open(out_path, "a", encoding="utf-8") as out:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(delay)
        except Exception as e:
            print(f" wiki batch error {e}")
            time.sleep(2)
    print(f" wiki: extracted {len(extracts)} new articles")
    return extracts

def build_processed():
    """
    Build processed .txt files from raw jsonl for training.
    Also creates combined input.txt
    """
    print("\n=== Building processed files ===")
    combined = []
    stats = {}

    # tononkira
    src = "tononkira"
    path = RAW_DIR / src / f"{src}.jsonl"
    texts = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    lyrics=rec.get("lyrics","").strip()
                    if lyrics:
                        # format: Title - Artist\nLyrics\n\n
                        t = rec.get("title","").strip()
                        a = rec.get("artist","").strip()
                        block = f"{t} - {a}\n{lyrics}\n"
                        texts.append(block)
                except: pass
    if texts:
        out = PROC_DIR / "tononkira.txt"
        with open(out, "w", encoding="utf-8") as o:
            o.write("\n\n".join(texts))
        stats["tononkira"] = {"count": len(texts), "chars": sum(len(t) for t in texts), "file": str(out)}
        combined.extend(texts)
        print(f"  tononkira: {len(texts)} lyrics, {sum(len(t) for t in texts)} chars")

    # vetso
    src="vetso"
    path=RAW_DIR/src/f"{src}.jsonl"
    texts=[]
    if path.exists():
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    poem=rec.get("poem","").strip()
                    if poem:
                        t=rec.get("title","").strip()
                        a=rec.get("author","").strip()
                        block=f"{t} - {a}\n{poem}\n"
                        texts.append(block)
                except: pass
    if texts:
        out=PROC_DIR/"vetso.txt"
        with open(out,"w",encoding="utf-8") as o:
            o.write("\n\n".join(texts))
        stats["vetso"]={"count":len(texts),"chars":sum(len(t) for t in texts),"file":str(out)}
        combined.extend(texts)
        print(f"  vetso: {len(texts)} poems")

    # ohabolana
    src="ohabolana"
    path=RAW_DIR/src/f"{src}.jsonl"
    texts=[]
    if path.exists():
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    p=rec.get("proverb","").strip()
                    if p:
                        texts.append(p)
                except: pass
    if texts:
        out=PROC_DIR/"ohabolana.txt"
        with open(out,"w",encoding="utf-8") as o:
            o.write("\n".join(texts))
        stats["ohabolana"]={"count":len(texts),"chars":sum(len(t) for t in texts),"file":str(out)}
        combined.extend(texts)
        print(f"  ohabolana: {len(texts)} proverbs")

    # takila
    src="takila"
    path=RAW_DIR/src/f"{src}.jsonl"
    texts=[]
    if path.exists():
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    c=rec.get("content","").strip()
                    if c:
                        t=rec.get("title","").strip()
                        block=f"{t}\n{c}\n"
                        texts.append(block)
                except: pass
    if texts:
        out=PROC_DIR/"takila.txt"
        with open(out,"w",encoding="utf-8") as o:
            o.write("\n\n".join(texts))
        stats["takila"]={"count":len(texts),"chars":sum(len(t) for t in texts),"file":str(out)}
        combined.extend(texts)
        print(f"  takila: {len(texts)} stories")

    # wikipedia
    path=RAW_DIR/"wikipedia"/"mg.wikipedia.jsonl"
    texts=[]
    if path.exists():
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    rec=json.loads(line)
                    ext=rec.get("extract","").strip()
                    if ext:
                        title=rec.get("title","").strip()
                        block=f"{title}\n{ext}\n"
                        texts.append(block)
                except: pass
    if texts:
        out=PROC_DIR/"wikipedia.txt"
        with open(out,"w",encoding="utf-8") as o:
            o.write("\n\n".join(texts))
        stats["wikipedia"]={"count":len(texts),"chars":sum(len(t) for t in texts),"file":str(out)}
        combined.extend(texts)
        print(f"  wikipedia: {len(texts)} articles")

    # combined input.txt for llm training (karpathy style one big file)
    if combined:
        # shuffle? For LLM training, keep order but ensure diverse mixing - we interleave by source already.
        # For reproducibility, just join with separator
        random.seed(42)
        random.shuffle(combined)
        out_combined = PROC_DIR / "input.txt"
        # Also create data/input.txt root for compatibility with llm-cpp docs (data/input.txt)
        root_input = Path(__file__).resolve().parent.parent / "data" / "input.txt"
        with open(out_combined,"w",encoding="utf-8") as o:
            o.write("\n\n".join(combined))
        with open(root_input,"w",encoding="utf-8") as o:
            o.write("\n\n".join(combined))
        stats["combined"]={"count":len(combined),"chars":sum(len(t) for t in combined),"file":str(out_combined),"root_file":str(root_input)}
        print(f"  combined: {len(combined)} docs, {sum(len(t) for t in combined)} chars -> {out_combined} and {root_input}")

        # also create stats
        total_chars = sum(len(t) for t in combined)
        total_words = sum(len(t.split()) for t in combined)
        stats["total"]={"chars":total_chars,"words":total_words,"docs":len(combined)}

    # write stats json
    with open(BASE_DIR / "stats.json","w",encoding="utf-8") as s:
        json.dump(stats, s, ensure_ascii=False, indent=2)
    # also write human readable txt stats
    with open(BASE_DIR / "STATS.md","w",encoding="utf-8") as s:
        s.write("# Malagasy Corpus Stats\n\n")
        for k,v in stats.items():
            s.write(f"## {k}\n```json\n{json.dumps(v, ensure_ascii=False, indent=2)}\n```\n\n")
    print(f"  stats written to {BASE_DIR / 'stats.json'}")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Scrape Malagasy corpus")
    parser.add_argument("--pages", type=int, default=5, help="pages per source to scrape (list pages)")
    parser.add_argument("--delay", type=float, default=1.0, help="delay between requests sec")
    parser.add_argument("--full", action="store_true", help="scrape full (overrides --pages, may take hours)")
    parser.add_argument("--skip-wiki", action="store_true", help="skip wikipedia")
    parser.add_argument("--wiki-articles", type=int, default=500, help="number of wiki articles")
    parser.add_argument("--sources", nargs="+", default=["tononkira","vetso","ohabolana","takila"], help="sources to scrape")
    args = parser.parse_args()

    ensure_dirs()
    session = requests.Session()

    pages_map = {}
    if args.full:
        for src in args.sources:
            pages_map[src] = SOURCES[src]["pages_total"]
    else:
        for src in args.sources:
            pages_map[src] = args.pages

    for src in args.sources:
        if src not in SOURCES:
            print(f"unknown source {src} skip")
            continue
        pages = pages_map[src]
        # cap takila full to 50 pages to be sane
        if src=="takila" and args.full:
            pages=50
        try:
            scrape_source(src, pages, args.delay, session)
        except Exception as e:
            print(f"Error scraping {src}: {e}")
            import traceback; traceback.print_exc()

    if not args.skip_wiki:
        try:
            scrape_wikipedia(num_articles=args.wiki_articles, delay=args.delay)
        except Exception as e:
            print(f"Error scraping wiki: {e}")
            import traceback; traceback.print_exc()

    stats = build_processed()
    print("\nDone. Stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
