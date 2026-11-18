import argparse

# Per-task BM25 retrieval thresholds used in the paper.
DEFAULT_RETRIEVAL_THRESHOLD = {
    "abstract_generation": 10.0,
    "review_writing": 80.0,
}

RB_STRATEGIES = ["global_lowest", "cluster_wise", "cluster_round", "global_random"]


def get_args(argv=None):
    p = argparse.ArgumentParser(description="SPRInG")

    p.add_argument("--stage", required=True, choices=["sft", "cl", "infer"],
                   help="sft: period-0 LoRA SFT | cl: continual learning + retrieval buffer | infer: generation")

    # model / task / data
    p.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
    p.add_argument("--task_name", type=str, default="abstract_generation",
                   choices=["abstract_generation", "review_writing"])
    p.add_argument("--profile_size", type=str, default="small", choices=["small", "large"])
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--prompt_path", type=str, default="./prompts/prompt.json")
    p.add_argument("--output_root", type=str, default=".",
                   help="Everything is written under <output_root>/runs/... (see spring/paths.py).")
    p.add_argument("--start_user", type=int, default=0)
    p.add_argument("--end_user", type=int, default=None, help="Exclusive. Default: all users.")
    p.add_argument("--seed", type=int, default=42)

    # LoRA / optimisation
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_epoch", type=int, default=2)
    p.add_argument("--cut_off", type=int, default=768)
    p.add_argument("--train_history_k", type=int, default=0,
                   help="Training only: #BM25 history items prepended to each example (0 in the paper).")
    p.add_argument("--overwrite", action="store_true",
                   help="Retrain/regenerate even if the output already exists (default: skip finished work).")

    # SFT stage
    p.add_argument("--sft_periods", type=str, default="0",
                   help="Comma-separated periods to SFT, or 'all'. SPRInG only needs period 0.")

    # continual learning: drift-aware selection
    p.add_argument("--selection_mode", choices=["bottom", "top", "random"], default="bottom")
    p.add_argument("--selection_p", type=int, default=30, help="Percent of each period's data used for training.")
    p.add_argument("--oppu_alpha", type=float, default=1.0,
                   help="Score = logp_adapter - (1+alpha) * logp_base. 'inf' -> -logp_base.")

    # retrieval buffer
    p.add_argument("--use_retrieval_buffer", action="store_true")
    p.add_argument("--retrieval_buffer_max", type=int, default=50)
    p.add_argument("--rb_overflow_strategy", choices=RB_STRATEGIES, default="global_lowest",
                   help="Strategy whose buffer is read at inference.")
    p.add_argument("--rb_build_strategies", type=str, default=",".join(RB_STRATEGIES),
                   help="Comma-separated buffers that --stage cl builds (default: all four). "
                        "cluster_wise/cluster_round need the SBERT model (--sbert_model_name).")
    p.add_argument("--rb_seed", type=int, default=42)
    p.add_argument("--cluster_k_max", type=int, default=10)
    p.add_argument("--sbert_model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--retrieval_threshold", type=float, default=None,
                   help="BM25 score threshold. Default: per-task value from the paper.")

    # inference
    p.add_argument("--final_rag_mode", type=str, default="interpolation",
                   choices=["vanilla", "plain_rb", "threshold", "interpolation"])
    p.add_argument("--k", type=int, default=2, help="Inference: max #retrieval-buffer items used (0 = all above threshold).")
    p.add_argument("--interp_lambda", type=float, default=0.5,
                   help="0.0 -> no-retrieval distribution only, 1.0 -> retrieval-augmented only.")
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--infer_batch_size", type=int, default=16)

    args = p.parse_args(argv)
    if args.retrieval_threshold is None:
        args.retrieval_threshold = DEFAULT_RETRIEVAL_THRESHOLD[args.task_name]
    args.rb_build_strategies = [s.strip() for s in args.rb_build_strategies.split(",") if s.strip()]
    unknown = [s for s in args.rb_build_strategies if s not in RB_STRATEGIES]
    if unknown or not args.rb_build_strategies:
        p.error(f"--rb_build_strategies: unknown or empty {unknown}; choose from {RB_STRATEGIES}")
    return args
