"""Split each user's history into 5 chronological periods and hold out queries.

Produces the format consumed by SPRInG (`final_dataset*.json`): a flat list with
5 consecutive entries per user (period 0 = oldest):

    {"user_name": "<name>_period<p>",
     "profile": [ ...training items, newest first... ],
     "query":   [ {"id", "input", "gold"}, ... ]}

Splitting rule (matches the released data):
  * sort the user's items by time (oldest -> newest), N items in total
  * periods 1-4 get ceil(N/5) items each, period 0 gets the remainder
  * inside each period items are ordered newest-first; the first ceil(10%)
    become queries, the rest the period's profile

Sources:
  --source longlamp  : HF `LongLaMP/LongLaMP`, config abstract_generation_user (train+val+test merged)
  --source jsonl     : output of preprocess/process_amazon_reviews.py

Example:
    python preprocess/split_periods.py --task abstract_generation --source longlamp \
        --out data/abstract_generation/final_dataset_small.json
"""
import argparse
import json
import math
import os
import uuid

TIME_KEY = {"abstract_generation": "year", "review_writing": "reviewTime"}


def to_query(task, item):
    if task == "abstract_generation":
        return {"id": item["id"], "input": item["title"], "gold": item["abstract"]}
    return {
        "id": item["id"],
        "input": (f"Generate a detailed review for a product '{item['product']}' "
                  f"with the following description: '{item['description']}'. "
                  f"The review has an overall rating of '{item['overall']}'. "
                  f"The review summary is: '{item['summary']}'."),
        "gold": item["reviewText"],
    }


def split_user(task, name, items, num_periods=5, test_ratio=0.1):
    key = TIME_KEY[task]
    for it in items:
        it.setdefault("id", uuid.uuid4().hex)
    # input lists are newest-first; reverse before a stable ascending sort
    chrono = sorted(reversed(items), key=lambda x: str(x.get(key, "")))
    n = len(chrono)
    per = math.ceil(n / num_periods)
    first = n - per * (num_periods - 1)
    if first <= 0:
        return []
    bounds = [0, first] + [first + per * i for i in range(1, num_periods)]
    out = []
    for p in range(num_periods):
        chunk = list(reversed(chrono[bounds[p]:bounds[p + 1]]))  # newest first
        cut = math.ceil(len(chunk) * test_ratio)
        out.append({
            "user_name": f"{str(name).replace(' ', '_')}_period{p}",
            "profile": chunk[cut:],
            "query": [to_query(task, it) for it in chunk[:cut]],
        })
    return out


def load_users(args):
    if args.source == "longlamp":
        from datasets import load_dataset
        ds = load_dataset("LongLaMP/LongLaMP", "abstract_generation_user")
        users = []
        for split in ("train", "val", "test"):
            d = ds[split].to_dict()
            name_col = next(c for c in ("name", "reviewerId", "author") if c in d)
            users += [{"name": n, "profile": p} for n, p in zip(d[name_col], d["profile"])]
        return users
    with open(args.input, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TIME_KEY))
    ap.add_argument("--source", required=True, choices=["longlamp", "jsonl"])
    ap.add_argument("--input", help="JSONL users file (for --source jsonl)")
    ap.add_argument("--out", required=True, help="Output path, e.g. data/<task>/final_dataset_small.json")
    ap.add_argument("--min_items", type=int, default=0)
    args = ap.parse_args()
    if args.source == "jsonl" and not args.input:
        ap.error("--input is required with --source jsonl")

    output = []
    for u in load_users(args):
        if len(u["profile"]) >= max(args.min_items, 5):
            output += split_user(args.task, u["name"], list(u["profile"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out} ({len(output) // 5} users)")


if __name__ == "__main__":
    main()
