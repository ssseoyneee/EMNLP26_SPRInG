"""Single source of truth for every file SPRInG reads or writes.

Everything produced by one configuration lives under one run directory:

<output_root>/runs/<task>/<profile_size>/<model>/<hparams>/seed<seed>/
    sft/user<u>_period0/                       period-0 LoRA adapter          (stage sft)
    cl/<selection>/user<u>_period<t>/          continual adapters, t = 1..4   (stage cl)
    rb/<selection>/<buffer>/user<u>.json       retrieval buffer per strategy  (stage cl)
    outputs/<selection>/<decoding><users>.json predictions                    (stage infer)
    configs/<stage><users>.json                arguments of every invocation

  <hparams>   lr0.0001_ep2_bs4_r8_a8       (+ _ga<n> / _cut<n> / _hist<k> when non-default)
  <selection> bottom30_alpha1              (+ _seed<n> for random selection)
  <buffer>    max50_global_lowest          (cluster strategies add _ck<k>, global_random adds _seed<n>)
  <decoding>  interp_lam0.5_k2_thr10_rp1.2 (vanilla / plain_rb / threshold likewise)
  <users>     _u0-10 when --start_user/--end_user restrict the range, empty otherwise

Every setting that changes a file's content appears in its path, so two
different configurations can never overwrite each other.
"""
import json
import os

ADAPTER_FILE = "adapter_model.safetensors"


def fmt(x):
    """Compact, stable number formatting: 1e-4 -> 0.0001, 1e-05 -> 1e-05, 1.0 -> 1, 0.5 -> 0.5."""
    return f"{x:g}" if isinstance(x, float) else str(x)


def model_tag(args):
    return args.model_name.rstrip("/").split("/")[-1]


def hparam_tag(args):
    tag = f"lr{fmt(args.learning_rate)}_ep{args.max_epoch}_bs{args.batch_size}_r{args.r}_a{args.alpha}"
    if args.gradient_accumulation_steps != 1:
        tag += f"_ga{args.gradient_accumulation_steps}"
    if args.cut_off != 768:
        tag += f"_cut{args.cut_off}"
    if args.train_history_k > 0:
        tag += f"_hist{args.train_history_k}"
    return tag


def selection_tag(args):
    tag = f"{args.selection_mode}{args.selection_p}_alpha{fmt(args.oppu_alpha)}"
    if args.selection_mode == "random":
        tag += f"_seed{args.seed}"
    return tag


def buffer_tag(args, strategy):
    tag = f"max{args.retrieval_buffer_max}_{strategy}"
    if strategy in ("cluster_wise", "cluster_round"):
        tag += f"_ck{args.cluster_k_max}"
    if strategy == "global_random":
        tag += f"_seed{args.rb_seed}"
    return tag


def decoding_tag(args):
    mode = args.final_rag_mode
    tag = {"interpolation": f"interp_lam{fmt(args.interp_lambda)}"}.get(mode, mode)
    if mode != "vanilla":
        tag += f"_k{args.k}"
    if mode in ("interpolation", "threshold"):
        tag += f"_thr{fmt(args.retrieval_threshold)}"
    if mode != "vanilla":
        tag += f"_{buffer_tag(args, args.rb_overflow_strategy)}"
    return tag + f"_rp{fmt(args.repetition_penalty)}"


def users_tag(args):
    if args.start_user == 0 and args.end_user is None:
        return ""
    return f"_u{args.start_user}-{'end' if args.end_user is None else args.end_user}"


# ------------------------------------------------------------------ directories / files
def run_dir(args):
    return os.path.join(args.output_root, "runs", args.task_name, args.profile_size,
                        model_tag(args), hparam_tag(args), f"seed{args.seed}")


def sft_adapter(args, user, period=0):
    return os.path.join(run_dir(args), "sft", f"user{user}_period{period}")


def cl_adapter(args, user, period):
    return os.path.join(run_dir(args), "cl", selection_tag(args), f"user{user}_period{period}")


def adapter_for_period(args, user, period):
    """Period 0 -> SFT adapter, periods 1-4 -> continually updated adapters."""
    return sft_adapter(args, user, 0) if period == 0 else cl_adapter(args, user, period)


def buffer_dir(args, strategy):
    return os.path.join(run_dir(args), "rb", selection_tag(args), buffer_tag(args, strategy))


def buffer_file(args, user, strategy):
    return os.path.join(buffer_dir(args, strategy), f"user{user}.json")


def prediction_file(args):
    return os.path.join(run_dir(args), "outputs", selection_tag(args), f"{decoding_tag(args)}{users_tag(args)}.json")


def adapter_exists(path):
    return os.path.isfile(os.path.join(path, ADAPTER_FILE))


def require_adapter(path, hint):
    if not adapter_exists(path):
        raise FileNotFoundError(f"Missing adapter: {path}\n  -> {hint}")
    return path


def save_config(args):
    """Record the exact arguments next to the artifacts they produced."""
    path = os.path.join(run_dir(args), "configs", f"{args.stage}{users_tag(args)}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    return path
