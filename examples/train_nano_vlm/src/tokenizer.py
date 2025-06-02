from transformers import AutoTokenizer

TOKENIZERS_CACHE = {}


def get_tokenizer(name: str) -> AutoTokenizer:
    if name not in TOKENIZERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]
