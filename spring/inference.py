"""Stage 3: generation for every user/period.

Period 0 uses the SFT adapter and plain decoding; periods 1-4 use the continually
updated adapters and (with --final_rag_mode interpolation) retrieval-interpolated decoding
over that period's retrieval buffer.
"""
import datetime
import gc
import json
import os

import torch
from peft import PeftModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from . import paths
from . import retrieval_buffer as rb
from .data import load_dataset_json, load_prompts, user_range
from .generate import interpolation_generate, retrieve_from_buffer
from .modeling import load_fresh_base, load_tokenizer
from .task_utils import build_query_prompt, get_output, name2taskid, process_text, split_batch

LONG_TASKS = ("review_writing", "abstract_generation")


def _check_inputs(args, users, uses_buffer):
    """Fail before loading any model if an adapter or buffer of this configuration is missing."""
    missing = []
    for user in users:
        for period in range(5):
            if not paths.adapter_exists(paths.adapter_for_period(args, user, period)):
                missing.append(paths.adapter_for_period(args, user, period))
        if uses_buffer and not os.path.isfile(paths.buffer_file(args, user, args.rb_overflow_strategy)):
            missing.append(paths.buffer_file(args, user, args.rb_overflow_strategy))
    if missing:
        listing = "\n  ".join(missing[:10]) + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
        raise FileNotFoundError(f"{len(missing)} required file(s) missing for run dir {paths.run_dir(args)}:\n  {listing}\n"
                                "  -> run --stage sft / --stage cl with the same settings first")


def inference(args):
    out_path = paths.prediction_file(args)
    if os.path.isfile(out_path) and not args.overwrite:
        print(f"[INFER] predictions already exist, skipping (use --overwrite): {out_path}")
        return
    uses_buffer = args.final_rag_mode != "vanilla"
    if uses_buffer and not args.use_retrieval_buffer:
        raise ValueError(f"--final_rag_mode {args.final_rag_mode} needs --use_retrieval_buffer")

    data = load_dataset_json(args)
    prompts = load_prompts(args)
    system_prompt = prompts[args.task_name]["system"]
    _check_inputs(args, user_range(args, len(data)), uses_buffer)
    tokenizer = load_tokenizer(args.model_name)  # native chat template at inference

    pred_all, adapter_path = [], None
    for user in tqdm(user_range(args, len(data)), desc="Inference users"):
        for period in range(5):
            idx = user * 5 + period
            adapter_path = paths.require_adapter(paths.adapter_for_period(args, user, period),
                                                 "run --stage sft / --stage cl with the same settings first")
            print(f"[INFER] user={user} period={period} adapter={adapter_path}")

            # A fresh base per user/period on purpose: swapping LoRA adapters on one shared
            # Gemma-3 base (load_adapter/set_adapter/delete_adapter) changed the logits
            # (max |dlogit| 1.3 on GPU), so it is NOT equivalent to the setting used for the paper.
            base_model = load_fresh_base(args.model_name)
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
            model.eval()

            buffer_texts = []
            if uses_buffer and period > 0:
                buffer_texts = rb.require_buffer(args, user, args.rb_overflow_strategy).get(f"period{period}", []) or []

            long_task = args.task_name in LONG_TASKS
            generation_config = {
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "max_new_tokens": 600 if long_task else 100,
                "repetition_penalty": args.repetition_penalty if long_task else 1.0,
                "use_cache": True,
            }

            batch_prompts, batch_ids = [], []
            for q in data[idx]["query"]:
                query_prompt = build_query_prompt(args.task_name, prompts, q["input"])

                if args.final_rag_mode == "interpolation" and period > 0:
                    text = interpolation_generate(query_prompt, buffer_texts, system_prompt, tokenizer,
                                                  model, args, generation_config)
                    text = get_output(text, args.task_name, args.model_name)
                    text = process_text(text)  # drop chat preamble / extra options / repetition
                    pred_all.append({"id": q["id"], "output": text})
                    print(f"ID:{q['id']}, Output: {text}")
                    continue

                retrieved = []
                if args.final_rag_mode == "plain_rb" and args.k > 0 and buffer_texts:
                    retrieved = BM25Okapi([d.split() for d in buffer_texts]).get_top_n(
                        query_prompt.split(), buffer_texts, n=args.k)
                elif args.final_rag_mode == "threshold":
                    retrieved = retrieve_from_buffer(query_prompt, buffer_texts, args.k, args.retrieval_threshold)

                chat = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": "\n".join(retrieved + [query_prompt])}]
                batch_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False))
                batch_ids.append(q["id"])

            if batch_prompts:
                outs = []
                with torch.no_grad():
                    for batch in split_batch(batch_prompts, args.infer_batch_size):
                        enc = tokenizer(batch, return_tensors="pt", padding=True,
                                        return_token_type_ids=False).to(model.device)
                        gen = model.generate(**enc, **generation_config)
                        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                        outs.extend(get_output(s, args.task_name, args.model_name) for s in decoded)
                for qid, text in zip(batch_ids, outs):
                    pred_all.append({"id": qid, "output": text})
                    print(f"ID:{qid}, Output: {text}")

            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    path = out_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "task": name2taskid[args.task_name],
        "golds": pred_all,
        "model": args.model_name,
        "run_dir": paths.run_dir(args),
        "selection": paths.selection_tag(args),
        "last_adapter": adapter_path,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval": {
            "use_retrieval_buffer": bool(args.use_retrieval_buffer),
            "final_rag_mode": args.final_rag_mode,
            "threshold": float(args.retrieval_threshold),
            "k": int(args.k),
            "overflow_strategy": args.rb_overflow_strategy,
            "interp_lambda": float(args.interp_lambda),
        },
        "args": vars(args),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Inference completed -> {path}")
