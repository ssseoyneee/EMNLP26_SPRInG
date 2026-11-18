"""Dataset loading and training-example construction.

NOTE: training examples (and therefore the retrieval-buffer texts, which are built
from them) are rendered with the plain-text GEMMA_CHAT_TEMPLATE below, not with
the tokenizer's native chat template. This matches the configuration used for the
reported results; do not change it if you want to reproduce them.
"""
import json
import os

from datasets import Dataset
from rank_bm25 import BM25Okapi

from .task_utils import TARGET_FIELD, get_first_k_tokens

GEMMA_CHAT_TEMPLATE = (
    "\n{% set bos_token = '<bos>' %}\n{% set eos_token = '<eos>' %}\n{{ bos_token }}\n"
    "{% if messages[0]['role'] == 'system' %}\nSystem: {{ messages[0]['content'].strip() }}\n"
    "{% set loop_messages = messages[1:] %}\n{% else %}\n{% set loop_messages = messages %}\n{% endif %}\n"
    "{% for message in loop_messages %}\n  {% if message['role'] == 'user' %}\n"
    "User: {{ message['content'].strip() }}\n  {% elif message['role'] == 'assistant' %}\n"
    "Assistant: {{ message['content'].strip() }}{{ eos_token }}\n  {% endif %}\n{% endfor %}\nAssistant:\n"
)


def load_dataset_json(args):
    path = os.path.join(args.data_path, args.task_name, f"final_dataset_{args.profile_size}.json")
    with open(path, "r") as f:
        return json.load(f)


def load_prompts(args):
    with open(args.prompt_path, "r") as f:
        return json.load(f)


def user_range(args, n_entries):
    """Entries are stored user-major: index = user * 5 + period."""
    n_users = n_entries // 5
    end = n_users if args.end_user is None else min(args.end_user, n_users)
    return range(args.start_user, end)


def construct_prompt(qdict, prompt_template, user_history, args):
    base_prompt = prompt_template[args.task_name]["OPPU_input"].format(**qdict)
    if args.train_history_k > 0 and user_history:
        processed = [{k: get_first_k_tokens(v, 768) if isinstance(v, str) else v for k, v in p.items()}
                     for p in user_history]
        history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in processed]
        bm25 = BM25Okapi([doc.split(" ") for doc in history_list])
        query_str = prompt_template[args.task_name]["retrieval_query"].format(**qdict)
        retrieved = bm25.get_top_n(query_str.split(" "), history_list, n=args.train_history_k)
        base_prompt = "\n".join(retrieved) + "\n" + base_prompt
    return base_prompt


def _render(tokenizer, model_name, messages, add_generation_prompt):
    if "gemma" in model_name.lower():
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def create_dataset(user_history_data, tokenizer, prompt_template, args, system_prompt):
    """One (prompt, chosen) pair per profile item; returns a HF Dataset of rendered strings."""
    data = []
    target = TARGET_FIELD[args.task_name]
    for idx, qdict in enumerate(user_history_data):
        qdict = {k: get_first_k_tokens(v, 768) if isinstance(v, str) else v for k, v in qdict.items()}
        prompt_str = construct_prompt(qdict, prompt_template, user_history_data[:idx], args)
        prompt_msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_str}]
        chosen_msgs = [{"role": "assistant", "content": qdict[target]}]
        data.append({
            "prompt": _render(tokenizer, args.model_name, prompt_msgs, add_generation_prompt=True),
            "chosen": _render(tokenizer, args.model_name, chosen_msgs, add_generation_prompt=False),
        })
    return Dataset.from_list(data)


def generate_and_tokenize_prompt(data_point, tokenizer, cut_off):
    tp = tokenizer(data_point["prompt"], truncation=True, max_length=cut_off, return_tensors=None, add_special_tokens=False)
    tc = tokenizer(data_point["chosen"], truncation=True, max_length=cut_off, return_tensors=None, add_special_tokens=False)
    return {
        "input_ids": tp["input_ids"] + tc["input_ids"],
        "attention_mask": tp["attention_mask"] + tc["attention_mask"],
        "labels": [-100] * len(tp["input_ids"]) + tc["input_ids"],
    }
