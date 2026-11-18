"""Per-user retrieval buffer: one slot per period (1-4), capacity `retrieval_buffer_max`.

File: spring.paths.buffer_file(...) ->
      {"period1": [...], "period2": [...], "period3": [...], "period4": [...]}
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch

from . import paths

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ------------------------------------------------------------------ storage
def _rb_path(args, user, strategy):
    return paths.buffer_file(args, user, strategy)


def _blank():
    return {f"period{i}": [] for i in range(1, 5)}


def load_buffer(args, user, strategy):
    path = _rb_path(args, user, strategy)
    if not os.path.isfile(path):
        return _blank()
    try:
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict) and all(f"period{i}" in obj for i in range(1, 5)):
            return obj
        if isinstance(obj, list):  # legacy flat list
            d = _blank()
            d["period1"] = obj
            return d
    except Exception:
        pass
    return _blank()


def _dedup(texts):
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def save_buffer_period(args, user, period, texts, strategy):
    d = load_buffer(args, user, strategy)
    d[f"period{period}"] = _dedup(texts)[: args.retrieval_buffer_max]
    os.makedirs(paths.buffer_dir(args, strategy), exist_ok=True)
    with open(_rb_path(args, user, strategy), "w") as fw:
        json.dump(d, fw, ensure_ascii=False, indent=2)


def has_period(args, user, period, strategy):
    return bool(load_buffer(args, user, strategy).get(f"period{period}"))


def require_buffer(args, user, strategy):
    path = _rb_path(args, user, strategy)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing retrieval buffer: {path}\n  -> run --stage cl with --use_retrieval_buffer "
                                f"using the same settings (see configs/ in the run directory).")
    return load_buffer(args, user, strategy)


def prev_period_texts(args, user, period, strategy):
    if period <= 1:
        return []
    return load_buffer(args, user, strategy).get(f"period{period - 1}", []) or []


# ------------------------------------------------------------------ overflow selection
def _clusters_by_score(labels, scores):
    cid2idxs = defaultdict(list)
    for i, c in enumerate(labels):
        cid2idxs[c].append(i)
    clusters = []
    for cid, idxs in cid2idxs.items():
        clusters.append({
            "cid": cid,
            "avg": float(np.mean([scores[i] for i in idxs])),
            "idxs_sorted": sorted(idxs, key=lambda i: scores[i]),
        })
    clusters.sort(key=lambda x: x["avg"])
    return clusters


def select_overflow_indices(strategy, capacity, labels, scores, seed=42):
    """Pick `capacity` of N candidates. Lower score = higher priority."""
    n = len(scores)
    all_idxs = list(range(n))
    if capacity >= n:
        return all_idxs

    if strategy == "global_lowest":
        return sorted(all_idxs, key=lambda i: scores[i])[:capacity]

    if strategy == "global_random":
        return np.random.default_rng(seed).choice(all_idxs, size=capacity, replace=False).tolist()

    if strategy in ("cluster_wise", "cluster_round"):
        clusters = _clusters_by_score(labels, scores)
        chosen = []
        if strategy == "cluster_wise":
            for c in clusters:
                for i in c["idxs_sorted"]:
                    chosen.append(i)
                    if len(chosen) >= capacity:
                        return chosen[:capacity]
            return chosen[:capacity]
        ptrs = [0] * len(clusters)
        while len(chosen) < capacity:
            advanced = False
            for k, c in enumerate(clusters):
                if ptrs[k] < len(c["idxs_sorted"]):
                    chosen.append(c["idxs_sorted"][ptrs[k]])
                    ptrs[k] += 1
                    advanced = True
                    if len(chosen) == capacity:
                        break
            if not advanced:
                break
        return chosen[:capacity]

    raise ValueError(f"Unknown overflow strategy: {strategy}")


def dynamic_k_labels(embeds, k_max, random_state=42):
    """K-means with K in [2, k_max] chosen by cosine silhouette."""
    n = len(embeds)
    if not SKLEARN_AVAILABLE or n < 3:
        return np.zeros(n, dtype=int), 1, None
    best_sil, best_labels, best_k = -1.0, None, 1
    for k in range(2, min(max(2, int(k_max)), n - 1) + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(embeds)
            sil = silhouette_score(embeds, labels, metric="cosine")
            if np.isfinite(sil) and sil > best_sil:
                best_sil, best_labels, best_k = sil, labels, k
        except Exception:
            continue
    if best_labels is None:
        return np.zeros(n, dtype=int), 1, None
    return best_labels, best_k, best_sil


CLUSTER_STRATEGIES = ("cluster_wise", "cluster_round")
_SBERT = {}


def load_sbert(args):
    """SBERT encoder for the cluster strategies, loaded once per process.
    Raises instead of silently falling back to global_lowest, which would write a
    global_lowest buffer under a cluster_* directory name."""
    hint = (f"  -> make '{args.sbert_model_name}' available (pip install sentence-transformers scikit-learn; "
            f"download the model once while online), or build only the other buffers, e.g. "
            f"--rb_build_strategies global_lowest,global_random")
    if not SBERT_AVAILABLE or not SKLEARN_AVAILABLE:
        raise RuntimeError("cluster_wise/cluster_round buffers need sentence-transformers and scikit-learn.\n" + hint)
    if args.sbert_model_name not in _SBERT:
        try:
            _SBERT[args.sbert_model_name] = SentenceTransformer(
                args.sbert_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            raise RuntimeError(f"Could not load SBERT model '{args.sbert_model_name}': {type(e).__name__}: {e}\n"
                               + hint) from e
    return _SBERT[args.sbert_model_name]


def embed(sbert, texts, batch_size=64):
    return sbert.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        show_progress_bar=False, normalize_embeddings=True)
