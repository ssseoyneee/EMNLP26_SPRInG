"""ROUGE-1 / ROUGE-L / BERTScore per (user, period), plus averages.

Example
-------
python evaluation/evaluate.py \
    --task_name abstract_generation --profile_size small \
    --pred "SPRInG=runs/abstract_generation/small/gemma-3-4b-it/<hparams>/seed42/outputs/bottom30_alpha1/interp_*.json" \
    --metrics rouge bertscore --out_dir eval_results

Scores per user-period follow the paper's protocol:
  * ROUGE: rouge_score BootstrapAggregator mid F-measure (use_stemmer=True)
  * BERTScore: roberta-large, no baseline rescaling, mean over that user-period's queries
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace("**", "*")
    return " ".join(s.split()).strip()


def load_preds(spec):
    """spec: one path, a comma-separated list, or a glob (e.g. user-sharded prediction files)."""
    # natural sort so user shards (…_u2-3 before …_u10-11) are read in user order
    files = sorted({f for part in spec.split(",") for f in (glob.glob(part) or [part])},
                   key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
    preds, n_conflict = {}, 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        items = obj if isinstance(obj, list) else next(
            (obj[k] for k in ("golds", "preds", "outputs", "results") if isinstance(obj.get(k), list)), [])
        for x in items:
            if x.get("id") is None:
                continue
            # A few query ids occur under two users (co-authored papers). As in the paper's
            # evaluation, the prediction that comes last (in user order) is used for both.
            if x["id"] in preds and preds[x["id"]] != x.get("output", ""):
                n_conflict += 1
            preds[x["id"]] = x.get("output", "")
    print(f"loaded {len(preds)} predictions from {len(files)} file(s)"
          + (f"; {n_conflict} repeated id(s) with different outputs, last one kept" if n_conflict else ""))
    return preds


def matched_pairs(entry, preds, normalize=False):
    out = []
    for q in entry["query"]:
        if q.get("id") in preds:
            p, r = preds[q["id"]], q["gold"]
            if normalize:
                p, r = normalize_text(p), normalize_text(r)
                if not p or not r:
                    continue
            out.append((p, r))
    return out


def rouge_scores(gold, preds, seed=0):
    from rouge_score import rouge_scorer, scoring
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    np.random.seed(seed)  # BootstrapAggregator resamples with np.random
    r1, rl = [], []
    for entry in tqdm(gold, desc="ROUGE", leave=False):
        pairs = matched_pairs(entry, preds)
        if not pairs:
            r1.append(np.nan); rl.append(np.nan)
            continue
        agg = scoring.BootstrapAggregator()
        for pred, ref in pairs:
            agg.add_scores(scorer.score(ref, pred))
        a = agg.aggregate()
        r1.append(round(a["rouge1"].mid.fmeasure, 4))
        rl.append(round(a["rougeL"].mid.fmeasure, 4))
    return {"rouge-1": r1, "rouge-l": rl}


def bertscore_scores(gold, preds, model_type, batch_size):
    import torch
    from bert_score import score as bscore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F = [], [], []
    for entry in tqdm(gold, desc="BERTScore", leave=False):
        pairs = matched_pairs(entry, preds, normalize=True)
        if not pairs:
            P.append(np.nan); R.append(np.nan); F.append(np.nan)
            continue
        cands, refs = [p for p, _ in pairs], [r for _, r in pairs]
        p, r, f = bscore(cands, refs, model_type=model_type, lang="en", verbose=False,
                         device=device, batch_size=batch_size, rescale_with_baseline=False)
        P.append(round(float(p.mean()), 4)); R.append(round(float(r.mean()), 4)); F.append(round(float(f.mean()), 4))
    return {"bertscore-p": P, "bertscore-r": R, "bertscore-f1": F}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--profile_size", required=True)
    ap.add_argument("--data_path", default="./data")
    ap.add_argument("--pred", nargs="+", required=True, help="NAME=path (one or more); path may be a glob or comma list to merge user shards")
    ap.add_argument("--metrics", nargs="+", default=["rouge"], choices=["rouge", "bertscore"])
    ap.add_argument("--bertscore_model", default="roberta-large")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--start_user", type=int, default=0)
    ap.add_argument("--end_user", type=int, default=None)
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--seed", type=int, default=0, help="Seed for ROUGE bootstrap aggregation")
    args = ap.parse_args()

    with open(os.path.join(args.data_path, args.task_name, f"final_dataset_{args.profile_size}.json")) as f:
        gold = json.load(f)
    n_users = len(gold) // 5
    end = n_users if args.end_user is None else min(args.end_user, n_users)
    gold = gold[args.start_user * 5: end * 5]
    users = np.repeat(np.arange(args.start_user, end), 5)
    periods = np.tile(np.arange(5), end - args.start_user)

    per_metric = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds = load_preds(path)
        res = {}
        if "rouge" in args.metrics:
            res.update(rouge_scores(gold, preds, args.seed))
        if "bertscore" in args.metrics:
            res.update(bertscore_scores(gold, preds, args.bertscore_model, args.batch_size))
        for metric, vals in res.items():
            per_metric.setdefault(metric, {"user": users, "period": periods})[name] = vals

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for metric, cols in per_metric.items():
        df = pd.DataFrame(cols)
        out = os.path.join(args.out_dir, f"{args.task_name}_{args.profile_size}_{metric}.csv")
        df.to_csv(out, index=False)
        methods = [c for c in df.columns if c not in ("user", "period")]
        for m in methods:
            rows.append({
                "metric": metric, "method": m,
                "avg_P1-4": round(df.loc[df.period > 0, m].mean(), 4),
                "avg_all": round(df[m].mean(), 4),
                **{f"P{p}": round(df.loc[df.period == p, m].mean(), 4) for p in range(5)},
            })
        print(f"saved {out}")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, f"{args.task_name}_{args.profile_size}_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
