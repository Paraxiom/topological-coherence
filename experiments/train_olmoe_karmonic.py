"""OLMoE-1B-7B: Karmonic spectral regularization for MoE expert balancing.

Applies Karmonic spectral filtering to expert routing distributions instead
of hidden states. Tests whether torus-aware regularization prevents expert
collapse better than standard auxiliary load balancing loss.

Model: allenai/OLMoE-1B-7B-0125 (64 experts, top-8, Apache 2.0)
       Also supports deepseek-ai/deepseek-moe-16b-base via --model flag

5 conditions:
  1. dpo_default_aux    — DPO + model's built-in auxiliary load balance loss
  2. dpo_no_aux         — DPO only, auxiliary loss disabled (measure collapse)
  3. dpo_sinkhorn_route — DPO + Sinkhorn OT on routing distributions
  4. dpo_karmonic_route — DPO + Karmonic spectral filtering on routing dists
  5. dpo_karmonic_both  — DPO + Karmonic on routing + Karmonic on hidden states

Usage:
    python -u train_olmoe_karmonic.py --condition all --output_dir results/olmoe
    python -u train_olmoe_karmonic.py --model deepseek-ai/deepseek-moe-16b-base
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from collections import defaultdict

import random as _rng
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Gradient scaling
# ---------------------------------------------------------------------------

class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(x, scale):
    return GradientScale.apply(x, scale)


# ---------------------------------------------------------------------------
# FourierTorusHead + KarmonicFilterLoss (same as prior experiments)
# ---------------------------------------------------------------------------

class FourierTorusHeadLLM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, torus_dim=2, n_modes=6):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),
        )
        self.register_buffer("modes", torch.arange(1, n_modes + 1).float())

    def forward(self, z):
        raw = self.net(z)
        angles = 2.0 * math.pi * torch.sigmoid(raw)
        n_angles = angles.unsqueeze(-1) * self.modes
        cos_vals = torch.cos(n_angles).permute(0, 2, 1)
        sin_vals = torch.sin(n_angles).permute(0, 2, 1)
        fourier = torch.stack([cos_vals, sin_vals], dim=-1)
        return angles, fourier.reshape(z.shape[0], -1)


class KarmonicFilterLoss(nn.Module):
    def __init__(self, torus_dim=2, n_modes=6, grid_size=12, t_uniformity=2.0):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.t = t_uniformity
        eigenvalues = [2.0 - 2.0 * math.cos(2.0 * math.pi * n / grid_size)
                       for n in range(1, n_modes + 1)]
        lam_1 = eigenvalues[0]
        lam_max = max(eigenvalues)
        weights = [(lam - lam_1) / (lam_max - lam_1) if lam_max > lam_1 else 0.0
                   for lam in eigenvalues]
        self.register_buffer("karmonic_weights", torch.tensor(weights))

    def forward(self, angles, fourier_embed):
        B = fourier_embed.shape[0]
        k = self.torus_dim
        m = self.n_modes
        if B < 2:
            return torch.tensor(0.0, device=fourier_embed.device)
        total = torch.tensor(0.0, device=fourier_embed.device)
        for n in range(m):
            start = 2 * k * n
            end = 2 * k * (n + 1)
            mode_slice = fourier_embed[:, start:end]
            sq_dists = torch.cdist(mode_slice, mode_slice, p=2).pow(2)
            mask = ~torch.eye(B, dtype=torch.bool, device=fourier_embed.device)
            neg_dists = -self.t * sq_dists
            neg_dists = neg_dists.masked_select(mask).view(B, B - 1)
            unif_n = torch.logsumexp(neg_dists, dim=1).mean() - math.log(B - 1)
            total = total + self.karmonic_weights[n] * unif_n
        return total


# ---------------------------------------------------------------------------
# Routing-specific Karmonic: applies spectral filter to expert assignments
# ---------------------------------------------------------------------------

class RoutingKarmonicHead(nn.Module):
    """Projects expert routing probabilities to torus coordinates.

    Input: (B, n_experts) routing probability vectors
    Output: angles on T^2 + Fourier embedding
    """
    def __init__(self, n_experts, hidden_dim=64, torus_dim=2, n_modes=6):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.net = nn.Sequential(
            nn.Linear(n_experts, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),
        )
        self.register_buffer("modes", torch.arange(1, n_modes + 1).float())

    def forward(self, routing_probs):
        raw = self.net(routing_probs)
        angles = 2.0 * math.pi * torch.sigmoid(raw)
        n_angles = angles.unsqueeze(-1) * self.modes
        cos_vals = torch.cos(n_angles).permute(0, 2, 1)
        sin_vals = torch.sin(n_angles).permute(0, 2, 1)
        fourier = torch.stack([cos_vals, sin_vals], dim=-1)
        return angles, fourier.reshape(routing_probs.shape[0], -1)


# ---------------------------------------------------------------------------
# Sinkhorn OT (for routing distributions)
# ---------------------------------------------------------------------------

def sinkhorn_ot_loss(current, reference=None, epsilon=0.1, n_iters=20):
    if reference is None:
        B, D = current.shape
        reference = F.normalize(torch.randn(B, D, device=current.device), dim=-1)
    x = F.normalize(current, dim=-1)
    y = F.normalize(reference, dim=-1)
    C = torch.cdist(x, y, p=2).pow(2)
    K = torch.exp(-C / epsilon)
    n, m = K.shape
    u = torch.ones(n, device=K.device) / n
    v = torch.ones(m, device=K.device) / m
    for _ in range(n_iters):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)
    T = torch.diag(u) @ K @ torch.diag(v)
    return (T * C).sum()


# ---------------------------------------------------------------------------
# Router hook infrastructure
# ---------------------------------------------------------------------------

class RouterHookManager:
    """Captures expert routing logits from all MoE layers."""

    def __init__(self):
        self.routing_logits = []  # list of (B*seq_len, n_experts) per layer
        self.handles = []
        self.enabled = True

    def _hook_fn(self, module, input, output):
        if not self.enabled:
            return
        # Router gates output logits before top-k selection
        # Different architectures expose this differently
        if isinstance(output, tuple):
            logits = output[0] if isinstance(output[0], torch.Tensor) else output[1]
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            return

        # Ensure we have (tokens, n_experts) shape
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.shape[-1])
        elif logits.dim() == 2:
            pass
        else:
            return

        self.routing_logits.append(logits)

    def register(self, model):
        """Find and hook all router/gate modules in the MoE model."""
        base = model
        for _ in range(5):
            if hasattr(base, "base_model"):
                base = base.base_model
            elif hasattr(base, "model"):
                base = base.model
            else:
                break

        n_hooks = 0
        for name, module in base.named_modules():
            # OLMoE: layers.X.mlp.gate (OlmoeTopKRouter — not nn.Linear!)
            # DeepSeek-MoE: layers.X.mlp.gate (nn.Linear)
            # Mixtral: layers.X.block_sparse_moe.gate (nn.Linear)
            # Match any module named "gate" inside mlp/moe/sparse paths
            is_gate = (
                name.endswith(".gate")
                and ("mlp" in name or "moe" in name or "sparse" in name)
            )
            if is_gate:
                handle = module.register_forward_hook(self._hook_fn)
                self.handles.append(handle)
                n_hooks += 1

        print(f"RouterHookManager: {n_hooks} gate hooks registered", flush=True)
        return n_hooks

    def get_and_clear(self):
        """Return captured routing logits and clear buffer."""
        data = self.routing_logits
        self.routing_logits = []
        return data

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Expert utilization metrics
# ---------------------------------------------------------------------------

def compute_routing_metrics(routing_logits_list, top_k=8):
    """Compute expert utilization metrics from captured routing logits.

    Returns dict with:
      - expert_entropy: entropy of aggregate expert usage (higher = more balanced)
      - max_expert_load: fraction of tokens going to most-used expert
      - min_expert_load: fraction going to least-used expert
      - load_std: std dev of expert load fractions
      - routing_saturation: fraction of experts receiving > 1% of tokens
      - expert_loads: per-expert load fractions (for heatmaps)
    """
    if not routing_logits_list:
        return {}

    # Aggregate routing decisions across all layers
    all_assignments = []
    n_experts = None
    for logits in routing_logits_list:
        if logits.shape[0] == 0:
            continue
        n_experts = logits.shape[-1]
        # Get top-k expert assignments
        _, indices = torch.topk(logits, min(top_k, n_experts), dim=-1)
        all_assignments.append(indices.reshape(-1))

    if not all_assignments or n_experts is None:
        return {}

    all_indices = torch.cat(all_assignments)
    counts = torch.bincount(all_indices, minlength=n_experts).float()
    total = counts.sum()
    if total == 0:
        return {}

    loads = counts / total
    # Entropy (max = log(n_experts) for uniform)
    loads_nonzero = loads[loads > 0]
    entropy = -(loads_nonzero * torch.log(loads_nonzero)).sum().item()
    max_entropy = math.log(n_experts)

    return {
        "expert_entropy": entropy,
        "expert_entropy_normalized": entropy / max_entropy if max_entropy > 0 else 0,
        "max_expert_load": loads.max().item(),
        "min_expert_load": loads.min().item(),
        "load_std": loads.std().item(),
        "routing_saturation": (loads > 0.01).sum().item() / n_experts,
        "n_experts": n_experts,
        "expert_loads": loads.cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

class PreferenceLoss(nn.Module):
    def __init__(self, model, tokenizer, beta=0.1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta

    def _get_answer_log_prob(self, question_text, answer_text):
        prompt = f"Q: {question_text}\nA:"
        full = f"Q: {question_text}\nA: {answer_text}"
        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        full_ids = self.tokenizer(full, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        outputs = self.model(full_ids, labels=full_ids.clone())
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        answer_tokens = full_ids[:, prompt_len:]
        n_answer = answer_tokens.shape[1]
        if n_answer == 0:
            return torch.tensor(0.0, device=device)
        token_log_probs = log_probs[0, prompt_len - 1:prompt_len - 1 + n_answer, :]
        answer_log_prob = token_log_probs.gather(
            1, answer_tokens[0, :n_answer].unsqueeze(1)
        ).squeeze(1).mean()
        return answer_log_prob

    def forward(self, question, correct_answer, wrong_answer):
        log_correct = self._get_answer_log_prob(question, correct_answer)
        log_wrong = self._get_answer_log_prob(question, wrong_answer)
        margin = log_correct - log_wrong
        loss = -F.logsigmoid(self.beta * margin)
        return loss, float(margin.detach())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _score_mc(model, tokenizer, question, answer, device):
    prompt = f"Q: {question}\nA:"
    full = f"Q: {question}\nA: {answer}"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        outputs = model(full_ids)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        answer_tokens = full_ids[:, prompt_len:]
        if answer_tokens.shape[1] == 0:
            return -100.0
        token_lp = log_probs[0, prompt_len - 1:-1, :]
        n_ans = answer_tokens.shape[1]
        token_lp = token_lp[:n_ans]
        ans_lp = token_lp.gather(1, answer_tokens[0, :n_ans].unsqueeze(1)).squeeze(1)
    return float(ans_lp.mean().cpu())


def eval_truthfulqa(model, tokenizer, max_samples=None):
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    device = next(model.parameters()).device
    model.eval()
    mc1_correct = 0
    mc2_scores = []
    for example in tqdm(dataset, desc="TruthfulQA"):
        q = example["question"]
        c1, l1 = example["mc1_targets"]["choices"], example["mc1_targets"]["labels"]
        c2, l2 = example["mc2_targets"]["choices"], example["mc2_targets"]["labels"]
        scores = [_score_mc(model, tokenizer, q, c, device) for c in c1]
        if int(np.argmax(scores)) == l1.index(1):
            mc1_correct += 1
        lp = np.array([_score_mc(model, tokenizer, q, c, device) for c in c2])
        probs = np.exp(lp - lp.max())
        probs /= probs.sum()
        mc2_scores.append(sum(probs[i] for i, l in enumerate(l2) if l == 1))
    n = len(dataset)
    return {"mc1_accuracy": mc1_correct / n, "mc1_correct": mc1_correct,
            "mc2_score": float(np.mean(mc2_scores)), "mc2_total": n}


def eval_perplexity(model, tokenizer, max_samples=50, max_length=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length * max_samples)
    input_ids = encodings.input_ids[0]
    device = next(model.parameters()).device
    model.eval()
    nlls = []
    for i in tqdm(range(0, min(len(input_ids) - 1, max_length * max_samples),
                        max_length), desc="Perplexity"):
        chunk = input_ids[i:i + max_length].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(chunk, labels=chunk.clone())
            nlls.append(outputs.loss.item())
    return {"perplexity": float(np.exp(np.mean(nlls))), "num_chunks": len(nlls)}


def eval_routing(model, tokenizer, router_hooks, max_samples=100, top_k=8):
    """Evaluate routing distribution on a sample of TruthfulQA prompts."""
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    device = next(model.parameters()).device
    model.eval()
    router_hooks.enabled = True

    all_routing_logits = []
    with torch.no_grad():
        for example in tqdm(dataset, desc="Routing eval"):
            router_hooks.get_and_clear()
            prompt = f"Q: {example['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            _ = model(**inputs)
            batch_logits = router_hooks.get_and_clear()
            all_routing_logits.extend(batch_logits)

    metrics = compute_routing_metrics(all_routing_logits, top_k=top_k)
    return metrics


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = {
    "dpo_default_aux":    {"dpo": True, "disable_aux": False, "ot_route": False,
                           "karmonic_route": False, "karmonic_hidden": False},
    "dpo_no_aux":         {"dpo": True, "disable_aux": True,  "ot_route": False,
                           "karmonic_route": False, "karmonic_hidden": False},
    "dpo_sinkhorn_route": {"dpo": True, "disable_aux": True,  "ot_route": True,
                           "karmonic_route": False, "karmonic_hidden": False},
    "dpo_karmonic_route": {"dpo": True, "disable_aux": True,  "ot_route": False,
                           "karmonic_route": True,  "karmonic_hidden": False},
    "dpo_karmonic_both":  {"dpo": True, "disable_aux": True,  "ot_route": False,
                           "karmonic_route": True,  "karmonic_hidden": True},
}


def load_training_data(tokenizer, max_samples=500, max_length=256):
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    prompts, questions, correct_answers, wrong_answers = [], [], [], []
    for example in dataset:
        q = example["question"]
        ids = tokenizer(f"Q: {q}\nA:", return_tensors="pt", truncation=True,
                        max_length=max_length)
        prompts.append(ids)
        questions.append(q)
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        correct_answers.append(choices[correct_idx].strip())
        wrong = [c.strip() for j, c in enumerate(choices) if j != correct_idx]
        wrong_answers.append(wrong)
    return prompts, questions, correct_answers, wrong_answers


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_condition(condition_name, config, args):
    out = Path(args.output_dir) / condition_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MODEL: {args.model}")
    print(f"Condition: {condition_name}")
    print(f"Config: {config}")
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}", flush=True)

    # Detect MoE config
    n_experts = getattr(model.config, "num_experts", None)
    if n_experts is None:
        n_experts = getattr(model.config, "num_local_experts", None)
    if n_experts is None:
        n_experts = getattr(model.config, "n_routed_experts", None)
    if n_experts is None:
        # Scan model for gate layers to infer n_experts
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "gate" in name.lower():
                n_experts = module.out_features
                break
    top_k = getattr(model.config, "num_experts_per_tok",
                    getattr(model.config, "num_experts_per_token", 8))

    print(f"MoE config: {n_experts} experts, top-{top_k} routing", flush=True)
    if n_experts is None:
        raise ValueError("Could not detect number of experts. Is this an MoE model?")

    # Disable auxiliary load balancing loss if requested
    orig_aux_coef = getattr(model.config, "router_aux_loss_coef", None)
    if config["disable_aux"]:
        if hasattr(model.config, "router_aux_loss_coef"):
            model.config.router_aux_loss_coef = 0.0
            print(f"Disabled aux loss (was {orig_aux_coef})", flush=True)
        elif hasattr(model.config, "aux_loss_alpha"):
            model.config.aux_loss_alpha = 0.0
            print("Disabled aux_loss_alpha", flush=True)
        else:
            print("WARNING: Could not find aux loss config to disable", flush=True)

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    dev = next(model.parameters()).device
    if torch.cuda.is_available():
        print(f"Device: {dev}, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Router hooks
    router_hooks = RouterHookManager()
    n_gate_hooks = router_hooks.register(model)
    if n_gate_hooks == 0:
        print("WARNING: No router gates found! Routing metrics will be empty.", flush=True)

    # Components
    dpo = PreferenceLoss(model=model, tokenizer=tokenizer, beta=args.dpo_beta)

    # Routing Karmonic head
    routing_torus = None
    routing_karmonic_fn = None
    if config["karmonic_route"]:
        routing_torus = RoutingKarmonicHead(
            n_experts=n_experts, hidden_dim=64,
            torus_dim=2, n_modes=args.n_modes
        ).to(dev)
        routing_karmonic_fn = KarmonicFilterLoss(
            torus_dim=2, n_modes=args.n_modes, grid_size=n_experts
        ).to(dev)

    # Hidden-state Karmonic head (for condition 5)
    hidden_torus = None
    hidden_karmonic_fn = None
    if config["karmonic_hidden"]:
        hidden_torus = FourierTorusHeadLLM(
            input_dim=hidden_size, hidden_dim=128,
            torus_dim=2, n_modes=args.n_modes
        ).to(dev)
        hidden_karmonic_fn = KarmonicFilterLoss(
            torus_dim=2, n_modes=args.n_modes, grid_size=12
        ).to(dev)

    # Hidden state hook (for karmonic_hidden condition)
    captured_hidden = [None]
    hidden_hook_handle = None
    if config["karmonic_hidden"]:
        def hidden_hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                captured_hidden[0] = output
            elif isinstance(output, tuple):
                captured_hidden[0] = output[0]
            else:
                captured_hidden[0] = getattr(output, "last_hidden_state", output)

        base = model
        for _ in range(5):
            if hasattr(base, "base_model"):
                base = base.base_model
            elif hasattr(base, "model"):
                base = base.model
            else:
                break
        if hasattr(base, "norm"):
            hidden_hook_handle = base.norm.register_forward_hook(hidden_hook_fn)
        elif hasattr(base, "final_layernorm"):
            hidden_hook_handle = base.final_layernorm.register_forward_hook(hidden_hook_fn)

    # Optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if routing_torus:
        trainable_params += list(routing_torus.parameters())
    if hidden_torus:
        trainable_params += list(hidden_torus.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Data
    print("Loading training data...", flush=True)
    prompts, questions, correct_answers, wrong_answers = load_training_data(
        tokenizer, max_samples=args.train_samples)
    print(f"Loaded {len(prompts)} training prompts", flush=True)

    # Pre-training routing metrics
    print("Measuring pre-training routing...", flush=True)
    pre_routing = eval_routing(model, tokenizer, router_hooks, max_samples=50, top_k=top_k)
    if pre_routing:
        print(f"Pre-training: entropy={pre_routing.get('expert_entropy_normalized', 0):.3f} "
              f"saturation={pre_routing.get('routing_saturation', 0):.2%} "
              f"max_load={pre_routing.get('max_expert_load', 0):.3f}", flush=True)

    # Training
    print(f"\nTraining {args.n_steps} steps...", flush=True)
    model.train()
    if routing_torus:
        routing_torus.train()
    if hidden_torus:
        hidden_torus.train()

    routing_buffer = []  # buffer of routing logit snapshots
    routing_buffer_size = 8
    hidden_buffer = []
    hidden_buffer_size = 8
    ref_routing_cache = None
    logs = []

    for step in tqdm(range(args.n_steps), desc="Training"):
        optimizer.zero_grad()
        idx = step % len(prompts)
        prompt_input = {k: v.to(dev) for k, v in prompts[idx].items()}
        attention_mask = prompt_input["attention_mask"]

        total_loss = torch.tensor(0.0, device=dev, requires_grad=True)
        log = {"step": step}

        # DPO
        wrong_ans = _rng.choice(wrong_answers[idx]) if wrong_answers[idx] else ""
        # Disable routing capture during DPO (two forward passes, noisy routing)
        router_hooks.enabled = False
        dpo_loss, margin = dpo(questions[idx], correct_answers[idx], wrong_ans)
        total_loss = total_loss + dpo_loss
        log["dpo_loss"] = float(dpo_loss.detach())
        log["preference_margin"] = margin

        # Forward for routing/hidden capture
        if config["karmonic_route"] or config["ot_route"] or config["karmonic_hidden"]:
            router_hooks.enabled = True
            router_hooks.get_and_clear()
            captured_hidden[0] = None

            outputs = model(**prompt_input)
            batch_routing = router_hooks.get_and_clear()

            # Aggregate routing logits from this step (all layers)
            if batch_routing:
                # OLMoE router outputs softmaxed probs; other archs may output raw logits.
                # Normalise to probs if not already (check if sums ≈ 1).
                all_probs = []
                for logits in batch_routing:
                    logits = logits.float()
                    if logits.sum(dim=-1).mean().item() < 0.95:
                        # Raw logits — apply softmax
                        logits = F.softmax(logits, dim=-1)
                    all_probs.append(logits.mean(dim=0))  # (n_experts,)
                # Average across layers → single (n_experts,) routing distribution
                step_routing = torch.stack(all_probs).mean(dim=0).unsqueeze(0)  # (1, n_experts)

                # Routing Karmonic
                if config["karmonic_route"] and routing_torus is not None:
                    routing_buffer.append(step_routing.detach())
                    if len(routing_buffer) >= routing_buffer_size:
                        context = torch.cat(routing_buffer, dim=0)
                        live_batch = torch.cat([context, step_routing], dim=0)
                        scaled = gradient_scale(live_batch, args.grad_scale)
                        angles, fourier = routing_torus(scaled)
                        rk_loss = routing_karmonic_fn(angles, fourier)
                        total_loss = total_loss + args.lambda_karmonic * rk_loss
                        log["routing_karmonic_loss"] = float(rk_loss.detach())
                        routing_buffer = []

                        # Log routing entropy for this mini-batch
                        with torch.no_grad():
                            avg_probs = context.mean(dim=0)
                            nz = avg_probs[avg_probs > 0]
                            log["buffer_routing_entropy"] = -(nz * torch.log(nz)).sum().item()
                    else:
                        log["routing_karmonic_loss"] = 0.0

                # Sinkhorn OT on routing
                if config["ot_route"]:
                    if ref_routing_cache is None:
                        ref_routing_cache = step_routing.detach().clone()
                    ot_loss = sinkhorn_ot_loss(step_routing, ref_routing_cache)
                    total_loss = total_loss + args.lambda_ot * ot_loss
                    log["ot_route_loss"] = float(ot_loss.detach())
                    ref_routing_cache = 0.99 * ref_routing_cache + 0.01 * step_routing.detach()

            # Hidden-state Karmonic (for condition 5)
            if config["karmonic_hidden"] and hidden_torus is not None and captured_hidden[0] is not None:
                hidden = captured_hidden[0]
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                hidden_buffer.append(pooled.detach())
                if len(hidden_buffer) >= hidden_buffer_size:
                    context = torch.cat(hidden_buffer, dim=0)
                    live_batch = torch.cat([context, pooled], dim=0)
                    scaled = gradient_scale(live_batch, args.grad_scale)
                    angles, fourier = hidden_torus(scaled)
                    hk_loss = hidden_karmonic_fn(angles, fourier)
                    total_loss = total_loss + args.lambda_karmonic * hk_loss
                    log["hidden_karmonic_loss"] = float(hk_loss.detach())
                    hidden_buffer = []
                else:
                    log["hidden_karmonic_loss"] = 0.0

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        log["total_loss"] = float(total_loss.detach())
        logs.append(log)

        if (step + 1) % args.log_every == 0:
            recent = logs[-args.log_every:]
            avg_loss = np.mean([l["total_loss"] for l in recent])
            msg = f"Step {step+1}/{args.n_steps}: loss={avg_loss:.4f}"
            msg += f" dpo={np.mean([l.get('dpo_loss',0) for l in recent]):.4f}"
            msg += f" margin={np.mean([l.get('preference_margin',0) for l in recent]):.3f}"
            if "routing_karmonic_loss" in recent[-1]:
                msg += f" rk={np.mean([l.get('routing_karmonic_loss',0) for l in recent]):.4f}"
            if "ot_route_loss" in recent[-1]:
                msg += f" ot={np.mean([l.get('ot_route_loss',0) for l in recent]):.4f}"
            if "hidden_karmonic_loss" in recent[-1]:
                msg += f" hk={np.mean([l.get('hidden_karmonic_loss',0) for l in recent]):.4f}"
            if "buffer_routing_entropy" in recent[-1]:
                msg += f" H={np.mean([l.get('buffer_routing_entropy',0) for l in recent]):.3f}"
            print(msg, flush=True)

    train_time = time.time() - t0

    with open(out / "training_logs.json", "w") as f:
        json.dump(logs, f)

    # Post-training evaluation
    print("\nEvaluating TruthfulQA...", flush=True)
    tq = eval_truthfulqa(model, tokenizer, max_samples=args.eval_samples)

    print("Evaluating perplexity...", flush=True)
    ppl = eval_perplexity(model, tokenizer)

    print("Evaluating post-training routing...", flush=True)
    post_routing = eval_routing(model, tokenizer, router_hooks, max_samples=100, top_k=top_k)

    eval_time = time.time() - t0 - train_time

    results = {
        "condition": condition_name,
        "config": config,
        "model": args.model,
        "n_experts": n_experts,
        "top_k": top_k,
        "n_steps": args.n_steps,
        "orig_aux_loss_coef": orig_aux_coef,
        "lambda_karmonic": args.lambda_karmonic if (config["karmonic_route"] or config["karmonic_hidden"]) else 0,
        "lambda_ot": args.lambda_ot if config["ot_route"] else 0,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        **tq, **ppl,
        "pre_routing": pre_routing,
        "post_routing": {k: v for k, v in post_routing.items() if k != "expert_loads"},
        "post_expert_loads": post_routing.get("expert_loads", []),
    }

    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    router_hooks.remove()
    if hidden_hook_handle:
        hidden_hook_handle.remove()

    print(f"\n{condition_name}:")
    print(f"  MC1={tq['mc1_accuracy']:.4f}  MC2={tq['mc2_score']:.4f}  PPL={ppl['perplexity']:.2f}")
    if post_routing:
        print(f"  Routing: entropy={post_routing.get('expert_entropy_normalized', 0):.3f} "
              f"saturation={post_routing.get('routing_saturation', 0):.2%} "
              f"max_load={post_routing.get('max_expert_load', 0):.3f} "
              f"load_std={post_routing.get('load_std', 0):.4f}")
    print(f"  Time: train={train_time:.0f}s eval={eval_time:.0f}s", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="OLMoE Karmonic Expert Balancing")
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--condition", type=str, default="all",
                        choices=list(CONDITIONS.keys()) + ["all"])
    parser.add_argument("--output_dir", type=str, default="results/olmoe")
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda_karmonic", type=float, default=0.01)
    parser.add_argument("--lambda_ot", type=float, default=0.01)
    parser.add_argument("--grad_scale", type=float, default=0.1)
    parser.add_argument("--n_modes", type=int, default=6)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    print(f"OLMoE Karmonic Expert Balancing")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.n_steps}")
    print(f"", flush=True)

    if args.condition == "all":
        conditions = list(CONDITIONS.keys())
    else:
        conditions = [args.condition]

    all_results = []
    for cond in conditions:
        r = run_condition(cond, CONDITIONS[cond], args)
        all_results.append(r)

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Condition':<24} {'MC1':>8} {'MC2':>8} {'PPL':>10} {'Entropy':>10} {'Saturation':>12} {'Train(s)':>10}")
    print(f"{'-'*90}")
    for r in all_results:
        post = r.get("post_routing", {})
        print(f"{r['condition']:<24} {r['mc1_accuracy']:>8.4f} {r['mc2_score']:>8.4f} "
              f"{r['perplexity']:>10.2f} "
              f"{post.get('expert_entropy_normalized', 0):>10.3f} "
              f"{post.get('routing_saturation', 0):>12.2%} "
              f"{r['train_time_s']:>10.1f}")

    combined = Path(args.output_dir) / "combined_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {combined}")


if __name__ == "__main__":
    main()
