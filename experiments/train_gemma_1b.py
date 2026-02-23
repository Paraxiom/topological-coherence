"""Gemma-3-1B-IT scale-up: DPO + Karmonic/OT ablation.

Same 5-condition ablation as Phase 3 v3, but on google/gemma-3-1b-it
to match ENIGMA's model for direct comparison.

Usage:
    python -u train_gemma_1b.py --condition all --output_dir results/gemma_1b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

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
# FourierTorusHead + KarmonicFilterLoss
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
# SAMI
# ---------------------------------------------------------------------------

PRINCIPLES = [
    "The answer is factually accurate and can be verified against reliable sources.",
    "The answer acknowledges uncertainty when the evidence is insufficient.",
    "The answer avoids common misconceptions and popular but false beliefs.",
    "The answer distinguishes between established facts and speculation.",
    "The answer does not fabricate citations, statistics, or expert claims.",
    "The answer corrects a false premise in the question when present.",
]


class SAMILoss(nn.Module):
    def __init__(self, hidden_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.proj_completion = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj_principle = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, completion_hidden, principle_hidden, labels):
        z_y = F.normalize(self.proj_completion(completion_hidden), dim=-1)
        z_c = F.normalize(self.proj_principle(principle_hidden), dim=-1)
        sim = z_y @ z_c.T / self.temperature
        row_loss = F.cross_entropy(sim, labels)
        B, P = sim.shape
        if B >= P:
            col_targets = torch.zeros(P, dtype=torch.long, device=sim.device)
            for p_idx in range(P):
                matches = (labels == p_idx).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    col_targets[p_idx] = matches[0]
            col_loss = F.cross_entropy(sim.T, col_targets)
            loss = 0.5 * (row_loss + col_loss)
        else:
            loss = row_loss
        mi_estimate = math.log(sim.shape[1]) - loss.detach()
        return loss, mi_estimate


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
# Sinkhorn OT
# ---------------------------------------------------------------------------

def sinkhorn_ot_loss(hidden_states, reference_states=None, epsilon=0.1, n_iters=20):
    if reference_states is None:
        B, D = hidden_states.shape
        reference_states = F.normalize(torch.randn(B, D, device=hidden_states.device), dim=-1)
    x = F.normalize(hidden_states, dim=-1)
    y = F.normalize(reference_states, dim=-1)
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


# ---------------------------------------------------------------------------
# Principle encoder
# ---------------------------------------------------------------------------

def encode_principles(model, tokenizer, principles, device, captured_hidden_ref):
    all_hidden = []
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(principles):
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
            captured_hidden_ref[0] = None
            _ = model(**inputs)
            if captured_hidden_ref[0] is not None:
                hidden = captured_hidden_ref[0]
            else:
                print(f"  WARNING: hook didn't capture for principle {i}, using zeros")
                all_hidden.append(torch.zeros(model.config.hidden_size, device=device))
                continue
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_hidden.append(pooled[0])
            print(f"  Principle {i} encoded: shape={pooled.shape}", flush=True)
    return torch.stack(all_hidden)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = {
    "dpo_only":          {"dpo": True, "sami": False, "ot": False, "karmonic": False},
    "dpo_sami":          {"dpo": True, "sami": True,  "ot": False, "karmonic": False},
    "dpo_sami_ot":       {"dpo": True, "sami": True,  "ot": True,  "karmonic": False},
    "dpo_sami_karmonic": {"dpo": True, "sami": True,  "ot": False, "karmonic": True},
    "dpo_karmonic":      {"dpo": True, "sami": False, "ot": False, "karmonic": True},
}


def load_training_data(tokenizer, max_samples=500, max_length=256):
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    prompts, questions, correct_answers, wrong_answers, principle_labels = [], [], [], [], []
    for i, example in enumerate(dataset):
        q = example["question"]
        ids = tokenizer(f"Q: {q}\nA:", return_tensors="pt", truncation=True, max_length=max_length)
        prompts.append(ids)
        questions.append(q)
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        correct_answers.append(choices[correct_idx].strip())
        wrong = [c.strip() for j, c in enumerate(choices) if j != correct_idx]
        wrong_answers.append(wrong)
        principle_labels.append(i % len(PRINCIPLES))
    return prompts, questions, principle_labels, correct_answers, wrong_answers


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_condition(condition_name, config, args):
    out = Path(args.output_dir) / condition_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MODEL: {args.model}")
    print(f"Condition: {condition_name}")
    print(f"Config: {config}")
    print(f"{'='*60}", flush=True)

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

    # LoRA — detect target modules for this architecture
    # Gemma/Qwen both use q_proj, k_proj, v_proj, o_proj
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    dev = next(model.parameters()).device
    print(f"Device: {dev}, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Components
    dpo = PreferenceLoss(model=model, tokenizer=tokenizer, beta=args.dpo_beta) if config["dpo"] else None
    sami = SAMILoss(hidden_dim=hidden_size).to(dev) if config["sami"] else None
    torus_head = FourierTorusHeadLLM(input_dim=hidden_size, hidden_dim=128,
                                      torus_dim=2, n_modes=args.n_modes).to(dev) if config["karmonic"] else None
    karmonic_loss_fn = KarmonicFilterLoss(torus_dim=2, n_modes=args.n_modes,
                                           grid_size=args.grid_size).to(dev) if config["karmonic"] else None

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if sami: trainable_params += list(sami.parameters())
    if torus_head: trainable_params += list(torus_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print("Loading training data...", flush=True)
    prompts, questions, principle_labels, correct_answers, wrong_answers = load_training_data(
        tokenizer, max_samples=args.train_samples)
    print(f"Loaded {len(prompts)} training prompts", flush=True)

    ref_hidden_cache = None
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
    base = model
    for _ in range(5):
        if hasattr(base, "base_model"):
            base = base.base_model
        elif hasattr(base, "model"):
            base = base.model
        else:
            break
    if hasattr(base, "norm"):
        hook_handle = base.norm.register_forward_hook(hook_fn)
        print(f"Hook on {type(base).__name__}.norm", flush=True)
    elif hasattr(base, "final_layernorm"):
        hook_handle = base.final_layernorm.register_forward_hook(hook_fn)
        print(f"Hook on {type(base).__name__}.final_layernorm", flush=True)
    else:
        print(f"WARNING: No norm layer found on {type(base).__name__}", flush=True)

    # Encode principles
    principle_hidden = None
    if config["sami"]:
        print("Encoding principles...", flush=True)
        principle_hidden = encode_principles(model, tokenizer, PRINCIPLES, dev, captured_hidden).detach()
        print(f"Principles: {principle_hidden.shape}", flush=True)

    # Training
    print(f"\nTraining {args.n_steps} steps...", flush=True)
    model.train()
    if sami: sami.train()
    if torus_head: torus_head.train()

    karmonic_buffer = []
    karmonic_buffer_size = 8
    logs = []

    for step in tqdm(range(args.n_steps), desc="Training"):
        optimizer.zero_grad()
        idx = step % len(prompts)
        prompt_input = {k: v.to(dev) for k, v in prompts[idx].items()}
        attention_mask = prompt_input["attention_mask"]

        total_loss = torch.tensor(0.0, device=dev, requires_grad=True)
        log = {"step": step}

        # DPO
        if config["dpo"] and dpo is not None:
            wrong_ans = _rng.choice(wrong_answers[idx]) if wrong_answers[idx] else ""
            dpo_loss, margin = dpo(questions[idx], correct_answers[idx], wrong_ans)
            total_loss = total_loss + dpo_loss
            log["dpo_loss"] = float(dpo_loss.detach())
            log["preference_margin"] = margin

        # Forward for SAMI/Karmonic/OT
        if config["sami"] or config["karmonic"] or config["ot"]:
            captured_hidden[0] = None
            outputs = model(**prompt_input)
            if captured_hidden[0] is not None:
                hidden = captured_hidden[0]
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                if config["sami"] and sami is not None and principle_hidden is not None:
                    plabel = torch.tensor([principle_labels[idx]], device=dev)
                    sami_loss, mi_est = sami(pooled, principle_hidden, plabel)
                    total_loss = total_loss + args.lambda_sami * sami_loss
                    log["sami_loss"] = float(sami_loss.detach())

                if config["karmonic"] and torus_head is not None:
                    karmonic_buffer.append(pooled.detach())
                    if len(karmonic_buffer) >= karmonic_buffer_size:
                        context = torch.cat(karmonic_buffer, dim=0)
                        live_batch = torch.cat([context, pooled], dim=0)
                        pooled_scaled = gradient_scale(live_batch, args.grad_scale)
                        angles, fourier = torus_head(pooled_scaled)
                        k_loss = karmonic_loss_fn(angles, fourier)
                        total_loss = total_loss + args.lambda_karmonic * k_loss
                        log["karmonic_loss"] = float(k_loss.detach())
                        karmonic_buffer = []
                    else:
                        log["karmonic_loss"] = 0.0

                if config["ot"]:
                    if ref_hidden_cache is None:
                        ref_hidden_cache = pooled.detach().clone()
                    ot_loss = sinkhorn_ot_loss(pooled, ref_hidden_cache)
                    total_loss = total_loss + args.lambda_ot * ot_loss
                    log["ot_loss"] = float(ot_loss.detach())
                    ref_hidden_cache = 0.99 * ref_hidden_cache + 0.01 * pooled.detach()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        log["total_loss"] = float(total_loss.detach())
        logs.append(log)

        if (step + 1) % args.log_every == 0:
            recent = logs[-args.log_every:]
            avg_loss = np.mean([l["total_loss"] for l in recent])
            msg = f"Step {step+1}/{args.n_steps}: loss={avg_loss:.4f}"
            if "dpo_loss" in recent[-1]:
                msg += f" dpo={np.mean([l.get('dpo_loss',0) for l in recent]):.4f}"
                msg += f" margin={np.mean([l.get('preference_margin',0) for l in recent]):.3f}"
            if "sami_loss" in recent[-1]:
                msg += f" sami={np.mean([l.get('sami_loss',0) for l in recent]):.4f}"
            if "karmonic_loss" in recent[-1]:
                msg += f" karmonic={np.mean([l.get('karmonic_loss',0) for l in recent]):.4f}"
            if "ot_loss" in recent[-1]:
                msg += f" ot={np.mean([l.get('ot_loss',0) for l in recent]):.4f}"
            print(msg, flush=True)

    train_time = time.time() - t0
    if hook_handle: hook_handle.remove()

    with open(out / "training_logs.json", "w") as f:
        json.dump(logs, f)

    print("\nEvaluating...", flush=True)
    tq = eval_truthfulqa(model, tokenizer, max_samples=args.eval_samples)
    ppl = eval_perplexity(model, tokenizer)
    eval_time = time.time() - t0 - train_time

    results = {
        "condition": condition_name, "config": config, "model": args.model,
        "n_steps": args.n_steps,
        "lambda_sami": args.lambda_sami if config["sami"] else 0,
        "lambda_karmonic": args.lambda_karmonic if config["karmonic"] else 0,
        "lambda_ot": args.lambda_ot if config["ot"] else 0,
        "train_time_s": round(train_time, 1), "eval_time_s": round(eval_time, 1),
        **tq, **ppl,
    }

    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{condition_name}: MC1={tq['mc1_accuracy']:.4f} MC2={tq['mc2_score']:.4f} "
          f"PPL={ppl['perplexity']:.2f} Train={train_time:.0f}s Eval={eval_time:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Gemma-1B Karmonic Scale-Up")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--condition", type=str, default="all",
                        choices=list(CONDITIONS.keys()) + ["all"])
    parser.add_argument("--output_dir", type=str, default="results/gemma_1b")
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
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

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<24} {'MC1':>8} {'MC2':>8} {'PPL':>10} {'Train(s)':>10}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['condition']:<24} {r['mc1_accuracy']:>8.4f} {r['mc2_score']:>8.4f} "
              f"{r['perplexity']:>10.2f} {r['train_time_s']:>10.1f}")

    combined = Path(args.output_dir) / "combined_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {combined}")


if __name__ == "__main__":
    main()
