"""Build per-user Amazon review histories (review_writing task).

Input : Amazon Review Data (2018) review file (.json / .json.gz, one JSON per line)
        and the matching metadata file.
Output: JSONL, one user per line: {"name", "input", "profile": [...], "output"}
        `profile` items: overall, reviewText, summary, reviewTime (YYYYMMDD),
        description, product. Newest review is moved to input/output.

Example:
    python preprocess/process_amazon_reviews.py \
        --reviews data/raw/All_Amazon_Review.json.gz \
        --meta data/raw/All_Amazon_Meta.json \
        --out data/raw/amazon_users.jsonl
"""
import argparse
import gzip
import json
from collections import OrderedDict, defaultdict
from datetime import datetime

from bs4 import BeautifulSoup
from tqdm import tqdm


def smart_open(path, mode="rt", encoding="utf-8"):
    with open(path, "rb") as f:
        head = f.read(2)
    return gzip.open(path, mode, encoding=encoding, errors="ignore") if head == b"\x1f\x8b" \
        else open(path, mode, encoding=encoding, errors="ignore")


def load_and_filter_reviews(path, min_review_words, min_summary_words):
    raw = []
    with smart_open(path) as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len((item.get("reviewText") or "").split()) < min_review_words:
                continue
            if len((item.get("summary") or "").split()) < min_summary_words:
                continue
            raw.append({k: item.get(k) for k in ("overall", "reviewerID", "reviewText", "summary", "asin", "reviewTime")})
    return raw


def build_meta_map(path, asin_set):
    meta = {}
    with smart_open(path) as f:
        for line in tqdm(f, desc="Building meta map"):
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            a, desc = m.get("asin"), m.get("description")
            if a in asin_set and desc:
                meta[a] = {"title": m.get("title"), "description": desc}
            if len(meta) == len(asin_set):
                break
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_review_words", type=int, default=120)
    ap.add_argument("--min_summary_words", type=int, default=4)
    args = ap.parse_args()

    raw = load_and_filter_reviews(args.reviews, args.min_review_words, args.min_summary_words)
    meta = build_meta_map(args.meta, {r["asin"] for r in raw})

    records = []
    for r in tqdm(raw, desc="Enrich/clean"):
        if r["asin"] not in meta:
            continue
        desc = meta[r["asin"]]["description"]
        if isinstance(desc, list):
            desc = " ".join(desc)
        r = dict(r, title=meta[r["asin"]]["title"],
                 description=BeautifulSoup(desc or "", "html.parser").get_text(" ", strip=True))
        r["reviewTime"] = datetime.strptime(r["reviewTime"], "%m %d, %Y").strftime("%Y%m%d")
        records.append(r)

    # de-duplicate by normalised review text (keep first)
    seen = {}
    for r in records:
        seen.setdefault(" ".join(r["reviewText"].split()), r)
    records = list(seen.values())

    by_user = defaultdict(list)
    for r in records:
        info = {k: v for k, v in r.items() if k not in ("reviewerID", "asin")}
        info["product"] = info.pop("title", None)
        by_user[r["reviewerID"]].append(info)

    with open(args.out, "w", encoding="utf-8") as fw:
        for rid, prof in tqdm(by_user.items(), desc="Writing users"):
            prof.sort(key=lambda x: x["reviewTime"], reverse=True)  # newest first
            latest = prof.pop(0)
            user = OrderedDict([
                ("name", rid),
                ("input", f"Generate a detailed review for a product '{latest['product']}' "
                          f"with the following description: '{latest['description']}'. "
                          f"The review has an overall rating of '{latest['overall']}'. "
                          f"The review summary is: '{latest['summary']}'."),
                ("profile", prof),
                ("output", latest["reviewText"]),
            ])
            fw.write(json.dumps(user, ensure_ascii=False).encode("utf-8", "surrogatepass").decode("utf-8") + "\n")
    print(f"saved {args.out} ({len(by_user)} users)")


if __name__ == "__main__":
    main()
