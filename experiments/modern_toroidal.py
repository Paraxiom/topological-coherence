#!/usr/bin/env python3
"""
TOROIDAL COHERENCE - Modern LLMs
================================
Real attention modification for Mistral/Qwen/OLMo.
Monkey-patches attention to inject toroidal bias before softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import json
import time
from datetime import datetime
import os
from functools import partial
from typing import Optional, Tuple

# ============================================================================
# TOROIDAL BIAS
# ============================================================================

def toroidal_distance(i: int, j: int, grid_size: int = 12) -> int:
    """Manhattan distance on 2D torus"""
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

_TOPO_CACHE = {}

def get_toroidal_bias(seq_len: int, grid_size: int = 12, radius: float = 2.0,
                      alpha: float = 0.5, device: str = 'cuda') -> torch.Tensor:
    """Get cached toroidal bias matrix"""
    key = (seq_len, grid_size, radius, alpha, device)
    if key not in _TOPO_CACHE:
        bias = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float16)
        for i in range(seq_len):
            for j in range(seq_len):
                dist = toroidal_distance(i, j, grid_size)
                if dist > radius:
                    bias[i, j] = -alpha * (dist - radius)
        _TOPO_CACHE[key] = bias
    return _TOPO_CACHE[key]

# ============================================================================
# ATTENTION PATCHING
# ============================================================================

TOROIDAL_ENABLED = False
TOROIDAL_CONFIG = {"grid_size": 12, "radius": 2.0, "alpha": 0.5}

def patched_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Patched attention that adds toroidal bias"""

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Apply rotary embeddings if available
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    elif hasattr(self, 'rotary_emb'):
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Repeat KV for GQA
    if self.num_key_value_heads != self.num_heads:
        n_rep = self.num_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(n_rep, dim=1)
        value_states = value_states.repeat_interleave(n_rep, dim=1)

    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # Apply causal mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # === TOROIDAL INJECTION ===
    if TOROIDAL_ENABLED:
        topo_bias = get_toroidal_bias(
            q_len,
            TOROIDAL_CONFIG["grid_size"],
            TOROIDAL_CONFIG["radius"],
            TOROIDAL_CONFIG["alpha"],
            attn_weights.device
        )
        # Add toroidal bias [seq, seq] -> [1, 1, seq, seq]
        attn_weights = attn_weights + topo_bias.unsqueeze(0).unsqueeze(0)
    # === END TOROIDAL ===

    # Softmax and apply to values
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotate half the hidden dims"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def patch_model_attention(model):
    """Monkey-patch all attention layers"""
    patched = 0
    for name, module in model.named_modules():
        if 'self_attn' in name and hasattr(module, 'q_proj'):
            # Store original forward
            module._original_forward = module.forward
            # Bind patched forward
            module.forward = patched_attention_forward.__get__(module, type(module))
            patched += 1
    print(f"Patched {patched} attention layers")
    return patched

def restore_model_attention(model):
    """Restore original attention"""
    for name, module in model.named_modules():
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward


# ============================================================================
# TEST PROMPTS
# ============================================================================

TEST_PROMPTS = [
    ("The capital of France is", ["Paris"]),
    ("Water freezes at", ["0", "32", "zero"]),
    ("The largest planet in our solar system is", ["Jupiter"]),
    ("Albert Einstein developed the theory of", ["relativity"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("World War II ended in the year", ["1945"]),
    ("The Mona Lisa was painted by", ["Leonardo", "Vinci"]),
    ("DNA stands for deoxyribonucleic", ["acid"]),
    ("The speed of light is approximately", ["300", "299", "186"]),
    ("William Shakespeare wrote the play", ["Hamlet", "Romeo", "Macbeth", "Othello"]),
    ("Mount Everest is located in", ["Nepal", "Himalaya"]),
    ("The atomic number of hydrogen is", ["1", "one"]),
    ("Photosynthesis converts sunlight into", ["energy", "glucose", "sugar"]),
    ("The currency of Japan is the", ["yen"]),
    ("Isaac Newton discovered", ["gravity", "laws of motion"]),
    ("The human body has how many chromosomes:", ["46", "forty-six"]),
    ("The Amazon River is located in", ["South America", "Brazil"]),
    ("The Great Wall of China was built during", ["Ming", "Qin", "dynasty"]),
    ("Oxygen makes up approximately what percent of air:", ["21", "20", "twenty"]),
    ("The Pythagorean theorem is about", ["triangle", "right angle", "hypotenuse"]),
]


# ============================================================================
# MAIN TEST
# ============================================================================

def run_test(model_name: str = "mistralai/Mistral-7B-v0.1", num_samples: int = 20):
    global TOROIDAL_ENABLED

    print("=" * 70)
    print("TOROIDAL COHERENCE TEST - MODERN LLMs")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Config: grid=12, radius=2.0, alpha=0.5")
    print("=" * 70)

    # Load
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Patch attention
    patch_model_attention(model)
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    prompts = (TEST_PROMPTS * ((num_samples // len(TEST_PROMPTS)) + 1))[:num_samples]

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "config": TOROIDAL_CONFIG.copy(),
        "baseline": {"correct": 0, "total": 0, "responses": []},
        "toroidal": {"correct": 0, "total": 0, "responses": []},
    }

    # === BASELINE ===
    print("\n[1/2] BASELINE (no constraint)...")
    TOROIDAL_ENABLED = False

    for i, (prompt, expected) in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)

        correct = any(e.lower() in resp.lower() for e in expected) if expected else True
        results["baseline"]["total"] += 1
        if correct:
            results["baseline"]["correct"] += 1
        results["baseline"]["responses"].append({"prompt": prompt, "response": resp[:150], "correct": correct})

        if i < 5:
            mark = "✓" if correct else "✗"
            print(f"  {mark} {resp[:70]}...")

    # === TOROIDAL ===
    print("\n[2/2] TOROIDAL (with constraint)...")
    TOROIDAL_ENABLED = True

    for i, (prompt, expected) in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)

        correct = any(e.lower() in resp.lower() for e in expected) if expected else True
        results["toroidal"]["total"] += 1
        if correct:
            results["toroidal"]["correct"] += 1
        results["toroidal"]["responses"].append({"prompt": prompt, "response": resp[:150], "correct": correct})

        if i < 5:
            mark = "✓" if correct else "✗"
            print(f"  {mark} {resp[:70]}...")

    # Results
    base_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    toro_acc = results["toroidal"]["correct"] / results["toroidal"]["total"]

    base_err = 1 - base_acc
    toro_err = 1 - toro_acc

    if base_err > 0:
        reduction = ((base_err - toro_err) / base_err) * 100
    else:
        reduction = 0 if toro_err == 0 else -100

    results["summary"] = {
        "baseline_accuracy": base_acc,
        "toroidal_accuracy": toro_acc,
        "baseline_error": base_err,
        "toroidal_error": toro_err,
        "error_reduction_percent": reduction,
    }

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy: {base_acc:.1%} ({results['baseline']['correct']}/{results['baseline']['total']})")
    print(f"Toroidal accuracy: {toro_acc:.1%} ({results['toroidal']['correct']}/{results['toroidal']['total']})")
    print(f"Error reduction:   {reduction:+.1f}%")
    print("=" * 70)

    # Save
    os.makedirs("./results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"./results/modern_toroidal_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    run_test(args.model, args.samples)
