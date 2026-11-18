"""Stage 1 (period-0 SFT) and Stage 2 (continual learning + retrieval buffer)."""
import gc
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from . import paths
from . import retrieval_buffer as rb
from .data import create_dataset, generate_and_tokenize_prompt, load_dataset_json, load_prompts, user_range
from .modeling import attach_lora, load_fresh_base, load_sft_base, load_tokenizer
from .selection import calculate_oppu_score, select_data


def _short(txt, n=120):
    s = "" if txt is None else str(txt).replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


def _training_args(args, out_dir):
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        num_train_epochs=args.max_epoch,
        save_strategy="no",
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        group_by_length=False,
        lr_scheduler_type="linear",
        report_to=[],
    )


def _collator(tokenizer):
    return DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ======================================================================
# Stage 1: period-0 SFT
# ======================================================================
def train_sft(args):
    data = load_dataset_json(args)
    prompts = load_prompts(args)
    system_prompt = prompts[args.task_name]["system"]
    periods = list(range(5)) if args.sft_periods == "all" else [int(p) for p in args.sft_periods.split(",")]

    tokenizer = load_tokenizer(args.model_name)
    base_model = load_sft_base(args.model_name, tokenizer)

    for user in tqdm(user_range(args, len(data)), desc="SFT users"):
        for period in periods:
            idx = user * 5 + period
            save_path = paths.sft_adapter(args, user, period)
            if paths.adapter_exists(save_path) and not args.overwrite:
                print(f"[SFT] skip user {user} period {period}: exists ({save_path})")
                continue
            print("#" * 70)
            print(f"[SFT] user {user} period {period} (entry {idx})")

            lora_config = LoraConfig(r=args.r, lora_alpha=args.alpha, target_modules=["q_proj", "v_proj"],
                                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            # The same base is re-wrapped for every user; PEFT re-initialises the LoRA weights.
            user_model = get_peft_model(base_model, lora_config)

            ds = create_dataset(data[idx]["profile"], tokenizer, prompts, args, system_prompt)
            ds = ds.shuffle()
            ds = ds.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, args.cut_off),
                        remove_columns=["prompt", "chosen"])
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            for name, module in user_model.named_modules():
                if "norm" in name:
                    module.to(user_model.dtype)

            trainer = Trainer(model=user_model, args=_training_args(args, save_path),
                              train_dataset=ds, data_collator=_collator(tokenizer))
            trainer.train()
            user_model.save_pretrained(save_path)
            print(f"[SFT] saved -> {save_path}")


# ======================================================================
# Stage 2: continual learning (periods 1..4) + retrieval buffer
# ======================================================================
def _update_retrieval_buffer(args, user, t, selected_raw, adapter_t, tokenizer):
    """Score S1-selected samples + previous buffer with (B, B+A_t) and keep the
    lowest-scoring (most drifted) texts. Buffers for all four overflow strategies
    are written so the strategy can be chosen at inference time.

    B+A_t is loaded fresh from the saved adapter and B is obtained with disable_adapter()
    (verified bitwise-identical to two separate models on GPU). Re-using the in-memory
    trained model instead is NOT equivalent (scores differed by up to 2.5e-2 on GPU)."""
    model_t = attach_lora(load_fresh_base(args.model_name), adapter_t, trainable=False)
    cache = {}

    def score(texts):
        out = []
        for txt in texts:
            if txt not in cache:
                cache[txt] = calculate_oppu_score(txt, model_t, tokenizer, alpha=args.oppu_alpha)
            out.append(cache[txt])
        return out

    s1_texts = [(it.get("prompt", "") or "") + (it.get("chosen", "") or "") for it in selected_raw]
    s1_scores = score(s1_texts)
    cap = int(args.retrieval_buffer_max)

    for strategy in args.rb_build_strategies:
        prev = rb.prev_period_texts(args, user, t, strategy)
        cand_texts = list(s1_texts) + list(prev)
        cand_scores = list(s1_scores) + score(prev)

        seen, uniq_texts, uniq_scores = set(), [], []
        for txt, sc in zip(cand_texts, cand_scores):
            if txt not in seen:
                seen.add(txt)
                uniq_texts.append(txt)
                uniq_scores.append(sc)

        if len(uniq_texts) <= cap:
            kept = list(range(len(uniq_texts)))
        elif strategy in rb.CLUSTER_STRATEGIES:
            labels, k, sil = rb.dynamic_k_labels(rb.embed(rb.load_sbert(args), uniq_texts), args.cluster_k_max)
            print(f"[RB][{strategy}] user{user} P{t} dynamic-K={k} silhouette={sil}")
            kept = rb.select_overflow_indices(strategy, cap, labels, uniq_scores, seed=args.rb_seed)
        else:
            kept = rb.select_overflow_indices(strategy, cap, None, uniq_scores, seed=args.rb_seed)

        kept_texts = [uniq_texts[i] for i in kept]
        if kept:
            print(f"[RB][{strategy}] user{user} P{t}: kept={len(kept)} "
                  f"S2_avg={float(np.mean([uniq_scores[i] for i in kept])):.4f} (S1={len(s1_texts)}, prevRB={len(prev)})")
        rb.save_buffer_period(args, user, t, kept_texts, strategy)

    del model_t
    _cleanup()


def _period_done(args, user, t):
    if not paths.adapter_exists(paths.cl_adapter(args, user, t)):
        return False
    if args.use_retrieval_buffer:
        return all(rb.has_period(args, user, t, s) for s in args.rb_build_strategies)
    return True


def continual_learning(args):
    """For t = 1..4:
         1) score period-t data with (base B, adapter A_{t-1}); keep the bottom-p% (S1)
         2) continue fine-tuning A_{t-1} on S1 -> A_t
         3) (optional) rescore S1 + previous buffer with (B, B+A_t) and refresh the buffer
    """
    data = load_dataset_json(args)
    prompts = load_prompts(args)
    system_prompt = prompts[args.task_name]["system"]
    tokenizer = load_tokenizer(args.model_name)
    if args.use_retrieval_buffer and any(s in rb.CLUSTER_STRATEGIES for s in args.rb_build_strategies):
        rb.load_sbert(args)  # fail before any training if the cluster buffers cannot be built
    print(f"[CL] retrieval buffers to build: {args.rb_build_strategies if args.use_retrieval_buffer else 'none'}")

    for user in tqdm(user_range(args, len(data)), desc="CL users"):
        if not args.overwrite and all(_period_done(args, user, t) for t in range(1, 5)):
            print(f"[CL] skip user {user}: all periods already done")
            continue
        print("=" * 80)
        print(f"Starting continual learning for User {user}")
        print("=" * 80)

        period_raw = []
        for p in range(5):
            ds = create_dataset(data[user * 5 + p]["profile"], tokenizer, prompts, args, system_prompt)
            ds = ds.shuffle()
            period_raw.append([{"prompt": pr, "chosen": ch} for pr, ch in zip(ds["prompt"], ds["chosen"])])
        print(f"[USER {user}] samples per period: {[len(x) for x in period_raw]}")

        a0 = paths.require_adapter(paths.sft_adapter(args, user, 0),
                                   "run --stage sft first with the same task/size/model/hparams/seed")
        adapters = [a0]

        for t in range(1, 5):
            print("#" * 70)
            print(f"User {user} - Period {t}")
            print("#" * 70)
            prev_adapter = adapters[t - 1]
            if not args.overwrite and _period_done(args, user, t):
                print(f"[CL] skip user {user} period {t}: adapter and buffer already exist")
                adapters.append(paths.cl_adapter(args, user, t))
                continue

            # 1) drift-aware selection: B+A_{t-1} vs B (same model, adapter disabled)
            score_m = attach_lora(load_fresh_base(args.model_name), prev_adapter, trainable=False)
            selected = select_data(period_raw[t], score_m, tokenizer,
                                   p_percent=args.selection_p / 100.0, alpha=args.oppu_alpha,
                                   mode=args.selection_mode, random_seed=args.seed)
            for i, it in enumerate(selected[:3]):
                print(f"  [S1][{i}] {_short(it['prompt'] + it['chosen'], 300)}")
            del score_m
            _cleanup()

            # 2) continue training A_{t-1} -> A_t
            train_base = load_fresh_base(args.model_name)
            train_base.config.use_cache = False
            try:
                train_base.gradient_checkpointing_enable()
            except Exception:
                pass
            train_model = attach_lora(train_base, prev_adapter, trainable=True)

            hf_ds = Dataset.from_list(selected)
            hf_ds = hf_ds.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, args.cut_off),
                              remove_columns=["prompt", "chosen"])
            hf_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            out_dir = paths.cl_adapter(args, user, t)
            os.makedirs(out_dir, exist_ok=True)
            trainer = Trainer(model=train_model, args=_training_args(args, out_dir),
                              train_dataset=hf_ds, data_collator=_collator(tokenizer))
            trainer.train()
            train_model.save_pretrained(out_dir)
            adapters.append(out_dir)
            print(f"[USER {user}] Period {t} adapter saved -> {out_dir}")

            del trainer, train_model, train_base
            _cleanup()

            # 3) retrieval buffer, scored with the saved A_t
            if args.use_retrieval_buffer:
                _update_retrieval_buffer(args, user, t, selected, out_dir, tokenizer)

        print(f"Completed continual learning for User {user}")
    print("All users completed!")
