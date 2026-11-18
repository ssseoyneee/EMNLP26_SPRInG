"""Model / tokenizer loading and on-disk layout of adapters."""
import os

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token())
    tok.pad_token = tok.eos_token
    return tok


def load_sft_base(model_name, tokenizer):
    """Base model for period-0 SFT: bf16 load -> prepare_model_for_kbit_training
    (upcasts non-quantised params to fp32, enables gradient checkpointing)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, token=hf_token()
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return prepare_model_for_kbit_training(model)


def load_fresh_base(model_name):
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token()
    )
    base.eval()
    return base


def attach_lora(base_model, adapter_path, trainable=False):
    m = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=trainable)
    if not trainable:
        m.eval()
    return m
