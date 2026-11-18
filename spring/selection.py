"""Drift-aware scoring and data selection.

Both the base model B and the adapted model B+A are evaluated with a single
PeftModel: B+A normally, B inside `model.disable_adapter()`. With unmerged LoRA
this gives exactly the same logits as a separately loaded base model, at half the
GPU memory and one model load instead of two.
"""
import math

import numpy as np
import torch


@torch.no_grad()
def calculate_oppu_score(text, model, tokenizer, alpha=1.0, max_length=512):
    """score = mean log p_{B+A}(x) - (1 + alpha) * mean log p_B(x).
    Low scores = samples the current adapter explains poorly relative to the base,
    i.e. evidence of preference drift. `model` must be an (unmerged) PeftModel in eval mode."""
    dev = model.get_input_embeddings().weight.device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    labels = enc["input_ids"][:, 1:]

    def mean_logp():
        logp = torch.log_softmax(model(**enc).logits, dim=-1)[:, :-1, :]
        return torch.gather(logp, -1, labels.to(logp.device).unsqueeze(-1)).squeeze(-1).mean()

    with model.disable_adapter():
        base_mean = mean_logp()
    if math.isinf(alpha):
        return (-base_mean).item()
    oppu_mean = mean_logp()
    return (oppu_mean - (1.0 + alpha) * base_mean).item()


def select_data(data, model, tokenizer, p_percent, alpha=1.0, mode="bottom", random_seed=None):
    scores = [
        calculate_oppu_score((it.get("prompt", "") or "") + (it.get("chosen", "") or ""),
                             model, tokenizer, alpha=alpha)
        for it in data
    ]
    n = len(data)
    k = max(1, int(round(n * p_percent)))
    if mode == "bottom":
        idxs = np.argsort(scores)[:k]
    elif mode == "top":
        idxs = np.argsort(scores)[-k:]
    elif mode == "random":
        if random_seed is not None:
            np.random.seed(random_seed)
        idxs = np.random.choice(n, size=k, replace=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Selected {k} samples ({mode} {p_percent * 100:.1f}%)")
    print(f"selected_indices={idxs.tolist()} (scores={[round(scores[i], 4) for i in idxs]})")
    return [data[i] for i in idxs]
