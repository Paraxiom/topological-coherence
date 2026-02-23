"""ENIGMA-Karmonic Hybrid: GRPO + SAMI + Karmonic Spectral Filter.

Synthesizes ENIGMA's three-layer geometry-aware training (arXiv:2510.11278)
with Paraxiom's Karmonic spectral regularization, replacing ENIGMA's generic
Sinkhorn OT with targeted toroidal spectral filtering.

Architecture:
    Qwen 2.5-0.5B + LoRA
      |
      +-- L_GRPO: Group-relative policy optimization (on-policy RL)
      |     - G completions per prompt, group-relative advantages
      |     - DR-GRPO (sequence-level ratio clipping, epsilon=0.1)
      |
      +-- L_SAMI: Symmetric InfoNCE (principle encoding)
      |     - Row: does completion identify its principle?
      |     - Column: does principle identify its completion?
      |     - Mutual information lower bound on I(Y;C|X)
      |
      +-- L_karmonic: Karmonic spectral filter (replaces Sinkhorn OT)
            - Hook last hidden layer, mean-pool
            - GradientScale(0.1) -> FourierTorusHead -> KarmonicFilterLoss
            - Mode-weighted uniformity (low modes preserved, high attenuated)

Five conditions:
  1. GRPO only           (baseline, matches ENIGMA ablation)
  2. GRPO + SAMI         (matches ENIGMA without OT)
  3. GRPO + SAMI + OT    (full ENIGMA reproduction)
  4. GRPO + SAMI + Karmonic  (our hypothesis: better than Sinkhorn)
  5. GRPO + Karmonic      (no SAMI, tests Karmonic alone with RL)

Usage:
    python train_enigma_karmonic.py --condition grpo_sami_karmonic --output_dir results/enigma_karmonic
    python train_enigma_karmonic.py --condition all --output_dir results/enigma_karmonic

References:
    - ENIGMA: arXiv:2510.11278 (Seneque et al., Oct 2025)
    - SAMI: arXiv:2404.14313 (Franken et al., 2024)
    - GRPO: arXiv:2402.03300 (DeepSeekMath)
    - Karmonic: Cormier 2026 (Zenodo)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    raise ImportError("pip install transformers datasets peft accelerate")


# ---------------------------------------------------------------------------
# Gradient scaling (from jepa-torus/src/train_v8b.py)
# ---------------------------------------------------------------------------

class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientScale.apply(x, scale)


# ---------------------------------------------------------------------------
# FourierTorusHead + KarmonicFilterLoss (from Phase 1)
# ---------------------------------------------------------------------------

class FourierTorusHeadLLM(nn.Module):
    def __init__(self, input_dim=896, hidden_dim=128, torus_dim=2, n_modes=6):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.embed_dim = 2 * torus_dim * n_modes
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
# SAMI: Symmetric Alignment via Mutual Information (InfoNCE)
# ---------------------------------------------------------------------------

# Constitutional principles for truthfulness alignment
# Adapted from ENIGMA's approach using Paraxiom's truthfulness domain
PRINCIPLES = [
    "The answer is factually accurate and can be verified against reliable sources.",
    "The answer acknowledges uncertainty when the evidence is insufficient.",
    "The answer avoids common misconceptions and popular but false beliefs.",
    "The answer distinguishes between established facts and speculation.",
    "The answer does not fabricate citations, statistics, or expert claims.",
    "The answer corrects a false premise in the question when present.",
]

NEGATIVE_PRINCIPLES = [
    "The answer presents a popular misconception as if it were true.",
    "The answer fabricates specific details to appear more authoritative.",
    "The answer confidently states something without evidence.",
    "The answer relies on a single unreliable source.",
]


class SAMILoss(nn.Module):
    """Symmetric Alignment via Mutual Information.

    Implements symmetric InfoNCE: maximizes I(Y; C | X) where
    Y = completion hidden states, C = constitutional principle embeddings.

    Row InfoNCE: given completion, identify the matching principle
    Col InfoNCE: given principle, identify the matching completion
    L_SAMI = 0.5 * (L_row + L_col)
    """

    def __init__(self, hidden_dim=896, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        # Project hidden states and principle embeddings to shared space
        self.proj_completion = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.proj_principle = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, completion_hidden, principle_hidden, labels):
        """Compute symmetric InfoNCE.

        Args:
            completion_hidden: (B, hidden_dim) mean-pooled completion hidden states
            principle_hidden: (P, hidden_dim) principle embeddings (P = num principles)
            labels: (B,) index into principles for each completion

        Returns:
            loss: scalar, symmetric InfoNCE loss
            mi_estimate: scalar, mutual information lower bound
        """
        # Project to shared space
        z_y = F.normalize(self.proj_completion(completion_hidden), dim=-1)  # (B, 128)
        z_c = F.normalize(self.proj_principle(principle_hidden), dim=-1)    # (P, 128)

        # Similarity matrix (B, P)
        sim = z_y @ z_c.T / self.temperature

        # Row InfoNCE: each completion -> find its principle
        row_loss = F.cross_entropy(sim, labels)

        # Column InfoNCE: each principle -> find its completions
        # Only valid when B >= P; with B=1 just use row loss
        B = sim.shape[0]
        P = sim.shape[1]
        if B >= P:
            # Build column targets: for each principle p, find which completion has label p
            col_targets = torch.zeros(P, dtype=torch.long, device=sim.device)
            for p_idx in range(P):
                matches = (labels == p_idx).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    col_targets[p_idx] = matches[0]
            col_loss = F.cross_entropy(sim.T, col_targets)
            loss = 0.5 * (row_loss + col_loss)
        else:
            # Not enough completions for meaningful column InfoNCE
            loss = row_loss

        # MI estimate (InfoNCE lower bound)
        mi_estimate = math.log(sim.shape[1]) - loss.detach()

        return loss, mi_estimate


# ---------------------------------------------------------------------------
# GRPO: Group Relative Policy Optimization
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Simplified GRPO for LoRA fine-tuning.

    Generates G completions per prompt, scores them, computes group-relative
    advantages, and updates via clipped policy gradient.

    Following DR-GRPO: sequence-level ratio clipping instead of token-level.
    """

    def __init__(
        self,
        model,
        tokenizer,
        ref_model=None,
        group_size=4,
        max_gen_len=256,
        temperature=1.0,
        clip_epsilon=0.1,
        beta_kl=0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model  # frozen reference for KL (optional)
        self.group_size = group_size
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl

    def _generate_completions(self, prompt_ids, attention_mask):
        """Generate G completions for a single prompt."""
        device = prompt_ids.device
        completions = []

        for _ in range(self.group_size):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_gen_len,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Extract only the generated part
            gen_ids = outputs[:, prompt_ids.shape[1]:]
            completions.append(gen_ids)

        return completions

    def _score_completion(self, prompt_ids, completion_ids):
        """Score a completion: simple format-based reward.

        Following ENIGMA's approach: reward = 1 if completion has
        reasoning structure (step-by-step), 0 otherwise.
        Can be extended with truthfulness-specific rewards.
        """
        text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True)

        reward = 0.0
        # Basic structure reward
        if len(text.strip()) > 10:
            reward += 0.1
        # Hedging/uncertainty acknowledgment (truthfulness signal)
        hedging_phrases = ["I think", "likely", "probably", "it's possible",
                           "evidence suggests", "according to", "however"]
        for phrase in hedging_phrases:
            if phrase.lower() in text.lower():
                reward += 0.1
                break
        # Penalize overconfident fabrication signals
        fabrication_signals = ["definitely", "100%", "everyone knows",
                               "it's obvious", "clearly"]
        for phrase in fabrication_signals:
            if phrase.lower() in text.lower():
                reward -= 0.1
                break

        return min(max(reward, 0.0), 1.0)

    def _compute_log_probs(self, model, prompt_ids, completion_ids):
        """Compute log probability of completion given prompt."""
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        labels = full_ids.clone()
        labels[:, :prompt_len] = -100  # mask prompt tokens

        outputs = model(input_ids=full_ids, labels=labels)
        # Return sequence-level log prob (normalized by length)
        log_prob = -outputs.loss  # negative CE = mean log prob per token
        return log_prob

    def compute_grpo_loss(self, prompt_ids, attention_mask):
        """Compute GRPO loss for one prompt.

        1. Generate G completions
        2. Score each
        3. Compute group-relative advantages
        4. Clipped policy gradient
        """
        device = prompt_ids.device
        completions = self._generate_completions(prompt_ids, attention_mask)

        # Score completions
        rewards = []
        for comp in completions:
            r = self._score_completion(prompt_ids, comp)
            rewards.append(r)

        rewards = torch.tensor(rewards, device=device)

        # Group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + eps)
        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)

        # Policy gradient with clipping
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid = 0

        for i, comp in enumerate(completions):
            if comp.shape[1] == 0:
                continue

            # Current policy log prob
            log_prob = self._compute_log_probs(self.model, prompt_ids, comp)

            # Reference log prob (for importance ratio)
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_log_prob = self._compute_log_probs(self.ref_model,
                                                           prompt_ids, comp)
            else:
                ref_log_prob = log_prob.detach()

            # Sequence-level importance ratio (DR-GRPO)
            ratio = torch.exp(log_prob - ref_log_prob)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - self.clip_epsilon,
                                        1.0 + self.clip_epsilon)

            # PPO-style surrogate
            adv = advantages[i]
            surrogate = torch.min(ratio * adv, clipped_ratio * adv)

            total_loss = total_loss - surrogate  # minimize negative reward
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return total_loss, float(rewards.mean()), float(advantages.abs().mean())


# ---------------------------------------------------------------------------
# Sinkhorn OT (ENIGMA's original regularizer, for comparison)
# ---------------------------------------------------------------------------

def sinkhorn_ot_loss(hidden_states, reference_states=None, epsilon=0.1,
                     n_iters=20):
    """Entropic optimal transport regularizer.

    Controls representation drift by penalizing Wasserstein distance
    between current and reference hidden state distributions.

    If no reference, uses uniform distribution.
    """
    if reference_states is None:
        # Compare to uniform distribution on unit sphere
        B, D = hidden_states.shape
        reference_states = F.normalize(torch.randn(B, D,
                                                    device=hidden_states.device), dim=-1)

    # Normalize
    x = F.normalize(hidden_states, dim=-1)
    y = F.normalize(reference_states, dim=-1)

    # Cost matrix (squared Euclidean distance)
    C = torch.cdist(x, y, p=2).pow(2)

    # Sinkhorn iterations
    K = torch.exp(-C / epsilon)
    n, m = K.shape
    u = torch.ones(n, device=K.device) / n
    v = torch.ones(m, device=K.device) / m

    for _ in range(n_iters):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)

    # Transport plan
    T = torch.diag(u) @ K @ torch.diag(v)

    # Wasserstein distance
    loss = (T * C).sum()
    return loss


# ---------------------------------------------------------------------------
# Evaluation (shared with Phase 1)
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
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                           split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    device = next(model.parameters()).device
    model.eval()
    mc1_correct = 0
    mc2_scores = []

    for example in tqdm(dataset, desc="TruthfulQA"):
        q = example["question"]
        c1 = example["mc1_targets"]["choices"]
        l1 = example["mc1_targets"]["labels"]
        c2 = example["mc2_targets"]["choices"]
        l2 = example["mc2_targets"]["labels"]

        # MC1
        scores = [_score_mc(model, tokenizer, q, c, device) for c in c1]
        if int(np.argmax(scores)) == l1.index(1):
            mc1_correct += 1

        # MC2
        lp = np.array([_score_mc(model, tokenizer, q, c, device) for c in c2])
        probs = np.exp(lp - lp.max())
        probs /= probs.sum()
        mc2_scores.append(sum(probs[i] for i, l in enumerate(l2) if l == 1))

    n = len(dataset)
    return {
        "mc1_accuracy": mc1_correct / n if n > 0 else 0,
        "mc1_correct": mc1_correct,
        "mc2_score": float(np.mean(mc2_scores)),
        "mc2_total": n,
    }


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


# ---------------------------------------------------------------------------
# Principle encoder
# ---------------------------------------------------------------------------

def encode_principles(model, tokenizer, principles, device, captured_hidden_ref):
    """Encode constitutional principles using the model's forward hook.

    Uses the same hook mechanism as training to capture last hidden layer.

    Args:
        captured_hidden_ref: mutable list [None] shared with hook_fn

    Returns: (P, hidden_dim) tensor
    """
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(principles):
            inputs = tokenizer(p, return_tensors="pt", truncation=True,
                               max_length=128).to(device)
            captured_hidden_ref[0] = None
            _ = model(**inputs)

            if captured_hidden_ref[0] is not None:
                hidden = captured_hidden_ref[0]
            else:
                # Fallback: use logits shape as indicator
                print(f"  WARNING: hook didn't capture for principle {i}, using zeros")
                hidden_dim = model.config.hidden_size
                all_hidden.append(torch.zeros(hidden_dim, device=device))
                continue

            # Mean pool
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_hidden.append(pooled[0])
            print(f"  Principle {i} encoded: shape={pooled.shape}", flush=True)

    return torch.stack(all_hidden)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

CONDITIONS = {
    "grpo_only":          {"grpo": True,  "sami": False, "ot": False, "karmonic": False},
    "grpo_sami":          {"grpo": True,  "sami": True,  "ot": False, "karmonic": False},
    "grpo_sami_ot":       {"grpo": True,  "sami": True,  "ot": True,  "karmonic": False},
    "grpo_sami_karmonic": {"grpo": True,  "sami": True,  "ot": False, "karmonic": True},
    "grpo_karmonic":      {"grpo": True,  "sami": False, "ot": False, "karmonic": True},
}


def load_training_data(tokenizer, max_samples=5000, max_length=256):
    """Load training prompts.

    Uses TruthfulQA questions as prompts (we're training for truthfulness,
    so the prompts should be truthfulness-relevant, unlike Phase 1's oasst1).
    """
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                           split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    principle_labels = []
    for i, example in enumerate(dataset):
        q = example["question"]
        prompt = f"Q: {q}\nA:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_length)
        prompts.append(ids)
        # Assign principle based on question category (round-robin for now)
        principle_labels.append(i % len(PRINCIPLES))

    return prompts, principle_labels


def run_condition(condition_name, config, args):
    """Run a single experimental condition."""
    out = Path(args.output_dir) / condition_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    print(f"Config: {config}")
    print(f"{'='*60}")

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dev = next(model.parameters()).device

    # Note: ref_model=None means GRPO uses detached log probs as reference
    # (avoids loading two models on single GPU which causes device_map hang)
    ref_model = None
    print(f"Model on device: {dev}, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Initialize components
    grpo = GRPOTrainer(
        model=model, tokenizer=tokenizer, ref_model=ref_model,
        group_size=args.group_size, max_gen_len=args.max_gen_len,
        temperature=1.0, clip_epsilon=0.1,
    ) if config["grpo"] else None

    sami = SAMILoss(hidden_dim=model.config.hidden_size).to(dev) if config["sami"] else None

    torus_head = FourierTorusHeadLLM(
        input_dim=model.config.hidden_size, hidden_dim=128,
        torus_dim=2, n_modes=args.n_modes,
    ).to(dev) if config["karmonic"] else None

    karmonic_loss_fn = KarmonicFilterLoss(
        torus_dim=2, n_modes=args.n_modes, grid_size=args.grid_size,
    ).to(dev) if config["karmonic"] else None

    # Optimizer over all trainable params
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if sami:
        trainable_params += list(sami.parameters())
    if torus_head:
        trainable_params += list(torus_head.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Load data
    print("Loading training data...")
    prompts, principle_labels = load_training_data(tokenizer, max_samples=args.train_samples)
    print(f"Loaded {len(prompts)} training prompts")

    # Reference hidden states for Sinkhorn OT (captured from base model)
    ref_hidden_cache = None

    # Hook for capturing hidden states
    captured_hidden = [None]

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            captured_hidden[0] = output
        elif isinstance(output, tuple):
            captured_hidden[0] = output[0]
        else:
            captured_hidden[0] = getattr(output, "last_hidden_state", output)

    # Register hook on final norm layer
    hook_handle = None
    print("Registering hook...")
    # Walk PEFT -> base model (max 5 hops to avoid infinite loop)
    base = model
    for _ in range(5):
        if hasattr(base, "base_model"):
            base = base.base_model
        elif hasattr(base, "model"):
            base = base.model
        else:
            break
    # Now find the norm layer
    if hasattr(base, "norm"):
        hook_handle = base.norm.register_forward_hook(hook_fn)
        print(f"Hook registered on {type(base).__name__}.norm")
    else:
        print(f"WARNING: No norm layer found on {type(base).__name__}, hook skipped")

    # Encode principles (must be after hook registration)
    principle_hidden = None
    if config["sami"]:
        print("Encoding constitutional principles...", flush=True)
        principle_hidden = encode_principles(
            model, tokenizer, PRINCIPLES, dev, captured_hidden
        ).detach()
        print(f"Principles encoded: {principle_hidden.shape}", flush=True)

    # Training loop
    import sys
    print(f"\nTraining for {args.n_steps} steps...", flush=True)
    sys.stdout.flush()
    model.train()
    if sami:
        sami.train()
    if torus_head:
        torus_head.train()

    logs = []
    for step in tqdm(range(args.n_steps), desc="Training"):
        optimizer.zero_grad()

        # Sample a prompt
        idx = step % len(prompts)
        prompt_input = {k: v.to(dev) for k, v in prompts[idx].items()}
        prompt_ids = prompt_input["input_ids"]
        attention_mask = prompt_input["attention_mask"]

        total_loss = torch.tensor(0.0, device=dev, requires_grad=True)
        log = {"step": step}

        # --- GRPO ---
        if config["grpo"] and grpo is not None:
            grpo_loss, mean_reward, mean_adv = grpo.compute_grpo_loss(
                prompt_ids, attention_mask
            )
            total_loss = total_loss + grpo_loss
            log["grpo_loss"] = float(grpo_loss.detach())
            log["mean_reward"] = mean_reward
            log["mean_advantage"] = mean_adv

        # Forward pass to capture hidden states for SAMI/Karmonic/OT
        if config["sami"] or config["karmonic"] or config["ot"]:
            captured_hidden[0] = None
            outputs = model(**prompt_input)
            ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=dev)

            if captured_hidden[0] is not None:
                hidden = captured_hidden[0]
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                # --- SAMI ---
                if config["sami"] and sami is not None and principle_hidden is not None:
                    plabel = torch.tensor([principle_labels[idx]], device=dev)
                    sami_loss, mi_est = sami(pooled, principle_hidden, plabel)
                    total_loss = total_loss + args.lambda_sami * sami_loss
                    log["sami_loss"] = float(sami_loss.detach())
                    log["mi_estimate"] = float(mi_est)

                # --- Karmonic ---
                if config["karmonic"] and torus_head is not None:
                    pooled_scaled = gradient_scale(pooled, args.grad_scale)
                    angles, fourier = torus_head(pooled_scaled)
                    k_loss = karmonic_loss_fn(angles, fourier)
                    total_loss = total_loss + args.lambda_karmonic * k_loss
                    log["karmonic_loss"] = float(k_loss.detach())

                # --- Sinkhorn OT ---
                if config["ot"]:
                    if ref_hidden_cache is None:
                        ref_hidden_cache = pooled.detach().clone()
                    ot_loss = sinkhorn_ot_loss(pooled, ref_hidden_cache)
                    total_loss = total_loss + args.lambda_ot * ot_loss
                    log["ot_loss"] = float(ot_loss.detach())
                    # Update reference cache (EMA)
                    ref_hidden_cache = 0.99 * ref_hidden_cache + 0.01 * pooled.detach()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        log["total_loss"] = float(total_loss.detach())
        logs.append(log)

        if (step + 1) % args.log_every == 0:
            recent = logs[-args.log_every:]
            avg_loss = np.mean([l["total_loss"] for l in recent])
            msg = f"Step {step+1}/{args.n_steps}: total_loss={avg_loss:.4f}"
            if "grpo_loss" in recent[-1]:
                avg_grpo = np.mean([l.get("grpo_loss", 0) for l in recent])
                msg += f" grpo={avg_grpo:.4f}"
            if "sami_loss" in recent[-1]:
                avg_sami = np.mean([l.get("sami_loss", 0) for l in recent])
                msg += f" sami={avg_sami:.4f}"
            if "karmonic_loss" in recent[-1]:
                avg_k = np.mean([l.get("karmonic_loss", 0) for l in recent])
                msg += f" karmonic={avg_k:.4f}"
            if "ot_loss" in recent[-1]:
                avg_ot = np.mean([l.get("ot_loss", 0) for l in recent])
                msg += f" ot={avg_ot:.4f}"
            print(msg, flush=True)

    train_time = time.time() - t0

    # Clean up hook
    if hook_handle:
        hook_handle.remove()

    # Save training logs
    with open(out / "training_logs.json", "w") as f:
        json.dump(logs, f)

    # Evaluate
    print("\nEvaluating...")
    tq = eval_truthfulqa(model, tokenizer, max_samples=args.eval_samples)
    ppl = eval_perplexity(model, tokenizer)
    eval_time = time.time() - t0 - train_time

    results = {
        "condition": condition_name,
        "config": config,
        "model": MODEL_NAME,
        "n_steps": args.n_steps,
        "lambda_sami": args.lambda_sami if config["sami"] else 0,
        "lambda_karmonic": args.lambda_karmonic if config["karmonic"] else 0,
        "lambda_ot": args.lambda_ot if config["ot"] else 0,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        **tq, **ppl,
    }

    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for {condition_name}:")
    print(f"  MC1: {tq['mc1_accuracy']:.4f}")
    print(f"  MC2: {tq['mc2_score']:.4f}")
    print(f"  PPL: {ppl['perplexity']:.2f}")
    print(f"  Train: {train_time:.0f}s, Eval: {eval_time:.0f}s")

    # Free GPU
    del model, ref_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    parser = argparse.ArgumentParser(description="ENIGMA-Karmonic Hybrid Training")
    parser.add_argument("--condition", type=str, default="all",
                        choices=list(CONDITIONS.keys()) + ["all"])
    parser.add_argument("--output_dir", type=str, default="results/enigma_karmonic")
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Training steps per condition")
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Max training prompts")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Max eval samples (None=all 817)")
    parser.add_argument("--group_size", type=int, default=4,
                        help="GRPO completions per prompt")
    parser.add_argument("--max_gen_len", type=int, default=128,
                        help="Max generation length")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda_sami", type=float, default=0.05)
    parser.add_argument("--lambda_karmonic", type=float, default=0.01)
    parser.add_argument("--lambda_ot", type=float, default=0.01)
    parser.add_argument("--grad_scale", type=float, default=0.1)
    parser.add_argument("--n_modes", type=int, default=6)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    if args.condition == "all":
        conditions = list(CONDITIONS.keys())
    else:
        conditions = [args.condition]

    all_results = []
    for cond in conditions:
        r = run_condition(cond, CONDITIONS[cond], args)
        all_results.append(r)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<24} {'MC1':>8} {'MC2':>8} {'PPL':>10} {'Train(s)':>10}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['condition']:<24} {r['mc1_accuracy']:>8.4f} {r['mc2_score']:>8.4f} "
              f"{r['perplexity']:>10.2f} {r['train_time_s']:>10.1f}")

    # Save combined
    combined = Path(args.output_dir) / "combined_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined}")


if __name__ == "__main__":
    main()
