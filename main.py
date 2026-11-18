import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import disable_progress_bar
from transformers import set_seed
from transformers.utils import logging

from spring.config import get_args
from spring.paths import run_dir, save_config


def main():
    args = get_args()
    set_seed(args.seed)
    disable_progress_bar()
    logging.set_verbosity_error()
    print(f"[{args.stage}] run dir: {run_dir(args)}")
    print(f"[{args.stage}] config  -> {save_config(args)}")

    if args.stage == "sft":
        from spring.train import train_sft
        train_sft(args)
    elif args.stage == "cl":
        from spring.train import continual_learning
        continual_learning(args)
    else:
        from spring.inference import inference
        inference(args)


if __name__ == "__main__":
    main()
