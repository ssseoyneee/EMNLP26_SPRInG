"""Retrieval-interpolated greedy decoding.

At each step the next-token distribution is
    p(y) = (1 - lambda) * p(y | x)  +  lambda * p(y | retrieved buffer items, x)
where both distributions come from the same personalised model.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers.cache_utils import DynamicCache


def retrieve_from_buffer(query, buffer_texts, k, threshold):
    """BM25 over the buffer; keep items with score >= threshold, top-k by score (all if k <= 0)."""
    if not buffer_texts:
        return []
    scores = BM25Okapi([d.split() for d in buffer_texts]).get_scores(query.split())
    idxs = [i for i, s in enumerate(scores) if s >= threshold]
    if k > 0:
        idxs = sorted(idxs, key=lambda i: scores[i], reverse=True)[:k]
    return [buffer_texts[i] for i in idxs]


def _sdpa_math_ctx():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            return nullcontext()


def interpolation_generate(query_prompt, buffer_texts, system_prompt, tokenizer, model, args, generation_config):
    """NOTE: like the original implementation, this caps generation_config["max_new_tokens"]
    in place to fit the context window; the caller re-creates the config per user."""
    base = getattr(model, "base_model", model)
    try:
        base.config.attn_implementation = "eager"
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    sdpa_ctx = _sdpa_math_ctx()

    device = model.get_input_embeddings().weight.device
    sw = int(getattr(getattr(base, "config", None), "sliding_window", 0) or 0)
    lam = max(0.0, min(1.0, float(args.interp_lambda)))

    retrieved = retrieve_from_buffer(query_prompt, buffer_texts, args.k, args.retrieval_threshold)

    def _join(parts):
        return "\n".join(p for p in parts if p)

    chat_v = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query_prompt}]
    chat_r = [{"role": "system", "content": system_prompt}, {"role": "user", "content": _join(retrieved + [query_prompt])}]
    ids_v = tokenizer.apply_chat_template(chat_v, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    ids_r = tokenizer.apply_chat_template(chat_r, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

    # keep inputs inside the (sliding-window) context
    want_new = generation_config.get("max_new_tokens", 100)
    effective_ctx = sw if sw > 0 else 1024
    max_input_len = max(512, min(effective_ctx - want_new - 10, effective_ctx - 10))
    ids_v = ids_v[:, -max_input_len:].contiguous()
    ids_r = ids_r[:, -max_input_len:].contiguous()

    # left-pad to a common length so both streams share cache positions
    L = max(ids_v.shape[1], ids_r.shape[1])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if ids_v.shape[1] < L:
        ids_v = F.pad(ids_v, (L - ids_v.shape[1], 0), value=pad_id).contiguous()
    if ids_r.shape[1] < L:
        ids_r = F.pad(ids_r, (L - ids_r.shape[1], 0), value=pad_id).contiguous()

    generation_config["max_new_tokens"] = max(1, min(want_new, effective_ctx - L - 1))

    if lam <= 1e-8 or lam >= 1.0 - 1e-8:
        input_ids = ids_v if lam <= 1e-8 else ids_r
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, **generation_config)
        return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    cache_v, cache_r = DynamicCache(), DynamicCache()
    gen_tokens = []
    penalised = None  # bool mask over the vocabulary: tokens generated so far
    eos_id = tokenizer.eos_token_id
    rep_penalty = generation_config.get("repetition_penalty", 1.0)
    max_new_tokens = generation_config["max_new_tokens"]

    try:
        with torch.no_grad(), sdpa_ctx:
            pos = torch.arange(L, device=device, dtype=torch.long)
            out_v = model(input_ids=ids_v, use_cache=True, past_key_values=cache_v, cache_position=pos)
            out_r = model(input_ids=ids_r, use_cache=True, past_key_values=cache_r, cache_position=pos)
            mix = (1.0 - lam) * F.softmax(out_v.logits[:, -1, :], dim=-1) + lam * F.softmax(out_r.logits[:, -1, :], dim=-1)
            next_tok = torch.argmax(mix, dim=-1).view(1, 1)
            penalised = torch.zeros(mix.shape[-1], dtype=torch.bool, device=mix.device)

            cur = L
            for _ in range(max_new_tokens):
                if eos_id is not None and int(next_tok.item()) == eos_id:
                    break
                gen_tokens.append(int(next_tok.item()))
                penalised[next_tok.view(-1)] = True
                if sw > 0 and cur >= sw:
                    print(f"[INFO] sliding-window limit reached at position {cur}; stopping.")
                    break
                pos = torch.tensor([cur], device=device, dtype=torch.long)
                out_v = model(input_ids=next_tok, use_cache=True, past_key_values=cache_v, cache_position=pos)
                out_r = model(input_ids=next_tok, use_cache=True, past_key_values=cache_r, cache_position=pos)
                mix = (1.0 - lam) * F.softmax(out_v.logits[:, -1, :], dim=-1) + lam * F.softmax(out_r.logits[:, -1, :], dim=-1)
                if rep_penalty > 1.0:
                    # each already-generated token divided once (same values as a per-token loop)
                    mix = torch.where(penalised, mix / rep_penalty, mix)
                next_tok = torch.argmax(mix, dim=-1).view(1, 1)
                cur += 1
    except Exception as e:
        print(f"[WARN] interpolation failed ({type(e).__name__}: {e}); falling back to retrieval-only decoding.")
        del cache_v, cache_r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            out = model.generate(input_ids=ids_r, **generation_config)
        return tokenizer.decode(out[0, ids_r.shape[1]:], skip_special_tokens=True).strip()

    del cache_v, cache_r
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
