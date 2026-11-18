# SPRInG: Continual LLM Personalization via Selective Parametric Adaptation and Retrieval-Interpolated Generation

[![Paper](https://img.shields.io/badge/EMNLP_2026-Paper-b31b1b)](https://aclanthology.org/2026.emnlp-main.1551/)
[![PDF](https://img.shields.io/badge/PDF-ACL_Anthology-blue)](https://aclanthology.org/2026.emnlp-main.1551.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ssseoyneee.github.io/spring_project_page/)

## Overview

Personalizing LLMs typically relies on static retrieval or one-time adaptation, assuming that user
preferences stay fixed. In real interactions, user interests keep evolving, so a personalized model
has to adapt to preference drift without catastrophic forgetting. Standard continual learning
updates indiscriminately on noisy interaction streams and cannot tell genuine preference shifts
from transient context.

**SPRInG** is a semi-parametric framework for continual personalization. Each user's history
arrives in chronological periods; a user-specific LoRA adapter is updated selectively on drift
signals, hard-to-learn residuals are kept in a replay buffer, and at inference the parametric
model is fused with retrieved history via probability interpolation.

## Key Mechanism

1. **Period-0 adaptation**: a user-specific LoRA adapter is trained on the user's first period.
2. **Drift-driven selective adaptation**: for every new period *t*, each interaction is scored with
   a likelihood-based novelty score

   `s(x) = log p_{B+A_{t-1}}(x) − (1+α)·log p_B(x)`

   (B: base model, A: user adapter). The lowest-scoring 30%, i.e. interactions the current
   adapter explains worst relative to the base model, are used to update `A_{t-1} → A_t`.
3. **Replay buffer**: the selected interactions and the previous buffer are re-scored with the
   updated adapter, and the most drift-indicative ones (up to 50 per period) are kept per user.
4. **Retrieval-interpolated generation**: at test time, buffer items are retrieved with BM25 and
   kept only above a per-task relevance threshold (strict gating). The next-token distributions with
   and without the retrieved items are mixed at every decoding step:

   `p(y) = (1−λ)·p(y | x) + λ·p(y | retrieved, x)`, with λ = 0.5.

## Installation

```bash
conda create -n spring python=3.10 -y
conda activate spring
pip install -r requirements.txt
export HF_TOKEN=...   # google/gemma-3-4b-it is a gated model
```

`requirements.txt` pins the versions used for the reported results (PyTorch 2.6.0, CUDA 12.4 build).

## Data

We use the [LongLaMP benchmark](https://longlamp-benchmark.github.io) (Hugging Face:
`LongLaMP/LongLaMP`, user setting) and split each user's history into five chronological periods.

Each task file is a flat list with five consecutive entries per user (period 0 = oldest):

```json
{"user_name": "<name>_period<p>",
 "profile": [ ...training interactions, newest first... ],
 "query":   [ {"id": "...", "input": "...", "gold": "..."} ]}
```

Place the files at `data/<task>/final_dataset_{small,large}.json`. For abstract generation,
`preprocess/split_periods.py` downloads LongLaMP (`abstract_generation_user`) and splits it directly:

```bash
python preprocess/split_periods.py --task abstract_generation --source longlamp \
    --out data/abstract_generation/final_dataset_small.json
```

For review writing, LongLaMP's product-review data has no review timestamps, so we take them from
[Amazon Reviews 2018](https://nijianmo.github.io/amazon/index.html): `preprocess/process_amazon_reviews.py`
rebuilds the per-user histories from the Amazon review and metadata files with LongLaMP's filtering
(reviews ≥ 120 words, summaries ≥ 4 words) while keeping `reviewTime`, and
`split_periods.py --task review_writing --source jsonl --input <users.jsonl>` splits them into periods.

## Usage

Run the full pipeline (period-0 adaptation → continual learning → inference):

```bash
bash scripts/run_spring.sh abstract_generation small 42
```

or stage by stage:

```bash
python main.py --stage sft   --task_name abstract_generation --profile_size small
python main.py --stage cl    --task_name abstract_generation --profile_size small --use_retrieval_buffer
python main.py --stage infer --task_name abstract_generation --profile_size small --use_retrieval_buffer
```

Evaluate (ROUGE-1/L and BERTScore per user and period):

```bash
python evaluation/evaluate.py --task_name abstract_generation --profile_size small \
    --pred "SPRInG=runs/abstract_generation/small/gemma-3-4b-it/*/seed42/outputs/*/interp_*.json" \
    --metrics rouge bertscore
```

Everything a configuration produces (adapters, buffers, predictions, the arguments of every
stage) is written to `runs/<task>/<size>/<model>/<hparams>/seed<seed>/`. Finished users and periods
are skipped, so an interrupted job can be resubmitted as is. Use `--start_user/--end_user` to split
users across GPUs.

Default hyper-parameters follow the paper:

| | value |
|---|---|
| base model | `google/gemma-3-4b-it` |
| LoRA | r = 8, α = 8, dropout 0.05, `q_proj`, `v_proj` |
| training | AdamW, lr 1e-4, 2 epochs, batch size 4, warmup 0.1, linear schedule, weight decay 0.01, grad-clip 0.3, bf16 |
| selection | bottom 30%, α = 1.0 |
| replay buffer | 50 items per period |
| decoding | greedy, max 600 new tokens, repetition penalty 1.2, λ = 0.5, top-k = 2 |
| BM25 threshold | abstract generation 10, review writing 80 |

Ablations: `--selection_mode {bottom,top,random}`, `--selection_p`, `--oppu_alpha`,
`--rb_overflow_strategy {global_lowest,cluster_wise,cluster_round,global_random}`, `--interp_lambda`,
`--final_rag_mode {vanilla,plain_rb,threshold,interpolation}`. The cluster-based buffer strategies
need the SBERT model given by `--sbert_model_name`.

## Repository Layout

```
main.py                 entry point (--stage sft | cl | infer)
spring/
  config.py             arguments and paper defaults
  paths.py              run-directory layout (every file read or written)
  data.py               dataset loading and prompt construction
  modeling.py           model / tokenizer loading
  selection.py          drift score and selective adaptation
  train.py              period-0 adaptation, continual learning, buffer update
  retrieval_buffer.py   replay buffer
  generate.py           retrieval-interpolated decoding
  inference.py          generation over all users and periods
  task_utils.py         task-specific parsing and output post-processing
prompts/prompt.json     task prompts
preprocess/             data preparation and chronological period split
evaluation/evaluate.py  ROUGE-1/L and BERTScore
scripts/run_spring.sh   full pipeline
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{kim2026spring,
    title={{SPRInG}: Continual {LLM} Personalization via Selective Parametric Adaptation and Retrieval-Interpolated Generation},
    author={Seoyeon Kim and Jaehyung Kim},
    booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
    year={2026},
    url={https://aclanthology.org/2026.emnlp-main.1551/}
}
```

## License

The code is released under the [MIT License](LICENSE). The Gemma-3 model and the datasets are
subject to their own licenses and terms of use.
