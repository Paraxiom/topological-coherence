#!/usr/bin/env python3
"""
TOKENIZER ANALYSIS
==================
Compare first 1440 tokens between Qwen and OLMo to understand
why limited bias works on Qwen but not OLMo.
"""
from transformers import AutoTokenizer
from collections import Counter

def analyze_tokenizer(model_name, n_tokens=1440):
    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Get first n_tokens
    tokens = []
    for i in range(min(n_tokens, vocab_size)):
        try:
            decoded = tokenizer.decode([i])
            tokens.append((i, decoded))
        except:
            tokens.append((i, f"<error:{i}>"))

    # Categorize tokens
    categories = {
        "special": [],      # <pad>, <eos>, etc.
        "punctuation": [],  # .,;:!?
        "numbers": [],      # 0-9
        "single_char": [],  # single letters
        "short_word": [],   # 2-4 char words
        "long_word": [],    # 5+ char words
        "subword": [],      # starts with special char like Ġ or ▁
        "byte": [],         # byte tokens
        "other": []
    }

    for i, tok in tokens:
        tok_clean = tok.strip()

        if tok.startswith("<") or tok.startswith("[") or i < 10:
            categories["special"].append((i, tok))
        elif tok_clean in ".,;:!?'\"-()[]{}":
            categories["punctuation"].append((i, tok))
        elif tok_clean.isdigit():
            categories["numbers"].append((i, tok))
        elif len(tok_clean) == 1 and tok_clean.isalpha():
            categories["single_char"].append((i, tok))
        elif len(tok_clean) <= 4 and tok_clean.isalpha():
            categories["short_word"].append((i, tok))
        elif len(tok_clean) > 4 and tok_clean.replace(" ", "").isalpha():
            categories["long_word"].append((i, tok))
        elif tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith(" "):
            categories["subword"].append((i, tok))
        elif len(tok) == 1 and ord(tok) < 256:
            categories["byte"].append((i, tok))
        else:
            categories["other"].append((i, tok))

    print(f"\nFirst {n_tokens} tokens breakdown:")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)}")
        if len(items) > 0 and len(items) <= 20:
            print(f"    Examples: {[t[1][:20] for t in items[:10]]}")

    # Show first 50 tokens
    print(f"\nFirst 50 tokens:")
    for i, tok in tokens[:50]:
        print(f"  {i:4d}: {repr(tok)[:30]}")

    # Show tokens 100-150
    print(f"\nTokens 100-150:")
    for i, tok in tokens[100:150]:
        print(f"  {i:4d}: {repr(tok)[:30]}")

    # Show tokens 1400-1440
    print(f"\nTokens 1400-1440:")
    for i, tok in tokens[1400:1440]:
        print(f"  {i:4d}: {repr(tok)[:30]}")

    return categories

if __name__ == "__main__":
    qwen_cats = analyze_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    olmo_cats = analyze_tokenizer("allenai/OLMo-1.7-7B-hf")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Category':<15} {'Qwen':>10} {'OLMo':>10}")
    print("-"*35)
    for cat in qwen_cats.keys():
        print(f"{cat:<15} {len(qwen_cats[cat]):>10} {len(olmo_cats[cat]):>10}")
