"""Karmonic-Constrained LLM Fine-Tuning for Hallucination Reduction.

Bridges V8b gradient-scaled Karmonic regularization (CIFAR-10) with
TLB hallucination reduction (TruthfulQA) via LoRA fine-tuning.

Architecture (mirrors V8b dual-path):
    Qwen 2.5-0.5B + LoRA
      tokens -> transformer -> hidden_states (B, seq_len, 896)
                                    |
                +-------------------+-------------------+
                |                                       |
          Path 1 (full grad)                  Path 2 (10% grad)
          LM head -> CE loss                  mean_pool -> (B, 896)
                                                  |
                                              GradientScale(0.1)
                                                  |
                                              FourierTorusHead(896->T^2)
                                                  |
                                              KarmonicFilterLoss

      total_loss = ce_loss + lambda_karmonic * karmonic_loss

Four conditions:
  1. Baseline       — no fine-tune, vanilla inference
  2. LoRA only      — LoRA fine-tune (CE only), vanilla inference
  3. LoRA+Karmonic  — LoRA + Karmonic loss, vanilla inference
  4. LoRA+Karmonic+TLB — LoRA + Karmonic loss, + TLB at inference

Usage:
    python train_karmonic_llm.py --condition baseline --output_dir results/karmonic_llm
    python train_karmonic_llm.py --condition all --output_dir results/karmonic_llm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    raise ImportError("pip install peft")


# ---------------------------------------------------------------------------
# Gradient scaling (from jepa-torus/src/train_v8b.py)
# ---------------------------------------------------------------------------

class GradientScale(torch.autograd.Function):
    """Scale gradients in backward pass without affecting forward pass."""

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
# Fourier Torus Head (adapted from jepa-torus/src/train_v5.py for LLM)
# ---------------------------------------------------------------------------

class FourierTorusHeadLLM(nn.Module):
    """Projects LLM hidden states to Fourier torus coordinates.

    Takes 896-dim hidden states (Qwen 2.5-0.5B) instead of 512-dim
    encoder outputs. Same Fourier expansion as jepa-torus V5.
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 128,
        torus_dim: int = 2,
        n_modes: int = 6,
    ):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.embed_dim = 2 * torus_dim * n_modes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm instead of BN for variable batch
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),
        )
        self.register_buffer("modes", torch.arange(1, n_modes + 1).float())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map pooled hidden states to Fourier torus coordinates.

        Args:
            z: Mean-pooled hidden states (B, input_dim).

        Returns:
            angles: Raw angles in [0, 2pi) of shape (B, torus_dim).
            fourier_embed: Fourier coordinates (B, 2*torus_dim*n_modes).
        """
        raw = self.net(z)
        angles = 2.0 * math.pi * torch.sigmoid(raw)

        # Fourier expansion grouped by mode
        n_angles = angles.unsqueeze(-1) * self.modes  # (B, k, m)
        cos_vals = torch.cos(n_angles).permute(0, 2, 1)  # (B, m, k)
        sin_vals = torch.sin(n_angles).permute(0, 2, 1)
        fourier = torch.stack([cos_vals, sin_vals], dim=-1)  # (B, m, k, 2)
        return angles, fourier.reshape(z.shape[0], -1)


# ---------------------------------------------------------------------------
# Karmonic Filter Loss (from jepa-torus/src/train_v5.py)
# ---------------------------------------------------------------------------

class KarmonicFilterLoss(nn.Module):
    """Mode-weighted uniformity implementing the Karmonic spectral filter.

    Low-frequency modes preserved (class discrimination),
    high-frequency modes attenuated toward uniformity (torus coverage).
    """

    def __init__(
        self,
        torus_dim: int = 2,
        n_modes: int = 6,
        grid_size: int = 12,
        t_uniformity: float = 2.0,
    ):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.t = t_uniformity

        eigenvalues = [
            2.0 - 2.0 * math.cos(2.0 * math.pi * n / grid_size)
            for n in range(1, n_modes + 1)
        ]
        lam_1 = eigenvalues[0]
        lam_max = max(eigenvalues)
        weights = [
            (lam - lam_1) / (lam_max - lam_1) if lam_max > lam_1 else 0.0
            for lam in eigenvalues
        ]
        self.register_buffer("karmonic_weights", torch.tensor(weights))

    def forward(self, angles: torch.Tensor, fourier_embed: torch.Tensor) -> torch.Tensor:
        """Compute Karmonic-filtered uniformity loss (scalar)."""
        B = fourier_embed.shape[0]
        k = self.torus_dim
        m = self.n_modes

        if B < 2:
            return torch.tensor(0.0, device=fourier_embed.device)

        total_uniformity = torch.tensor(0.0, device=fourier_embed.device)
        for n in range(m):
            start = 2 * k * n
            end = 2 * k * (n + 1)
            mode_slice = fourier_embed[:, start:end]

            sq_dists = torch.cdist(mode_slice, mode_slice, p=2).pow(2)
            mask = ~torch.eye(B, dtype=torch.bool, device=fourier_embed.device)
            neg_dists = -self.t * sq_dists
            neg_dists = neg_dists.masked_select(mask).view(B, B - 1)
            unif_n = torch.logsumexp(neg_dists, dim=1).mean() - math.log(B - 1)

            total_uniformity = total_uniformity + self.karmonic_weights[n] * unif_n

        return total_uniformity


# ---------------------------------------------------------------------------
# Toroidal Logit Processor (from topological-coherence logit_bias.py)
# ---------------------------------------------------------------------------

class ToroidalLogitProcessor:
    """Distance-based toroidal logit processor for HuggingFace generate().

    Maps token IDs onto an N x N torus, biases logits toward tokens
    topologically close to recent context.
    """

    def __init__(self, grid_size: int = 12, radius: float = 2.0, alpha: float = 0.3,
                 context_window: int = 32, bias_strength: float = 1.0):
        self.grid_size = grid_size
        self.radius = radius
        self.alpha = alpha
        self.context_window = context_window
        self.bias_strength = bias_strength
        self._n_positions = grid_size * grid_size
        # Precompute distance matrix for the torus
        self._dist = self._build_distance_matrix()

    def _build_distance_matrix(self) -> np.ndarray:
        n = self.grid_size
        pos = self._n_positions
        idx = np.arange(pos)
        x = idx % n
        y = idx // n
        dx = np.abs(x[:, None] - x[None, :])
        dy = np.abs(y[:, None] - y[None, :])
        dx = np.minimum(dx, n - dx)
        dy = np.minimum(dy, n - dy)
        return dx + dy

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        context_len = min(self.context_window, input_ids.shape[1])
        if context_len == 0:
            return scores

        recent_ids = input_ids[:, -context_len:]
        recent_positions = (recent_ids % self._n_positions).cpu().numpy()

        bias = torch.zeros_like(scores)
        for b in range(batch_size):
            pos_counts = np.zeros(self._n_positions, dtype=np.float32)
            for pos in recent_positions[b]:
                pos_counts[pos] += 1.0

            total_weight = pos_counts.sum()
            if total_weight > 0:
                torus_bias = np.zeros(self._n_positions, dtype=np.float32)
                for p in range(self._n_positions):
                    if pos_counts[p] > 0:
                        for q in range(self._n_positions):
                            d = self._dist[p, q]
                            if d <= self.radius:
                                torus_bias[q] += pos_counts[p] * self.bias_strength
                            else:
                                torus_bias[q] += pos_counts[p] * self.bias_strength * np.exp(
                                    -self.alpha * (d - self.radius)
                                )
                torus_bias /= total_weight
                vocab_positions = np.arange(vocab_size) % self._n_positions
                vocab_bias = torus_bias[vocab_positions]
                vocab_bias -= vocab_bias.mean()
                bias[b] = torch.from_numpy(vocab_bias).to(scores.device)

        return scores + bias


# ---------------------------------------------------------------------------
# Evaluation (from atlas-llm/atlas_bias/eval.py)
# ---------------------------------------------------------------------------

def _score_completion(model, tokenizer, question, answer,
                      logits_processor, device) -> float:
    prompt = f"Q: {question}\nA:"
    full = f"Q: {question}\nA: {answer}"

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

        if logits_processor is not None:
            for t in range(prompt_len, logits.shape[1]):
                prefix = full_ids[:, :t]
                logits[:, t - 1, :] = logits_processor(prefix, logits[:, t - 1, :])

        log_probs = torch.log_softmax(logits, dim=-1)
        answer_tokens = full_ids[:, prompt_len:]

        if answer_tokens.shape[1] == 0:
            return -100.0

        token_log_probs = log_probs[0, prompt_len - 1:-1, :]
        n_answer = answer_tokens.shape[1]
        token_log_probs = token_log_probs[:n_answer]

        answer_log_probs = token_log_probs.gather(
            1, answer_tokens[0, :n_answer].unsqueeze(1)
        ).squeeze(1)

    return float(answer_log_probs.mean().cpu())


def eval_truthfulqa_mc1(model, tokenizer, logits_processor=None,
                        max_samples=None) -> dict:
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                           split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    device = next(model.parameters()).device
    model.eval()

    for example in tqdm(dataset, desc="TruthfulQA MC1"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)

        scores = []
        for choice in choices:
            score = _score_completion(model, tokenizer, question, choice,
                                      logits_processor, device)
            scores.append(score)

        if int(np.argmax(scores)) == correct_idx:
            correct += 1
        total += 1

    return {"mc1_accuracy": correct / total if total > 0 else 0.0,
            "mc1_correct": correct, "mc1_total": total}


def eval_truthfulqa_mc2(model, tokenizer, logits_processor=None,
                        max_samples=None) -> dict:
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                           split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    scores_list = []
    device = next(model.parameters()).device
    model.eval()

    for example in tqdm(dataset, desc="TruthfulQA MC2"):
        question = example["question"]
        choices = example["mc2_targets"]["choices"]
        labels = example["mc2_targets"]["labels"]

        log_probs = []
        for choice in choices:
            score = _score_completion(model, tokenizer, question, choice,
                                      logits_processor, device)
            log_probs.append(score)

        log_probs = np.array(log_probs)
        probs = np.exp(log_probs - log_probs.max())
        probs = probs / probs.sum()
        correct_prob = sum(probs[i] for i, l in enumerate(labels) if l == 1)
        scores_list.append(correct_prob)

    return {"mc2_score": float(np.mean(scores_list)),
            "mc2_total": len(scores_list)}


def eval_perplexity(model, tokenizer, max_samples=100, max_length=512) -> dict:
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
        target = chunk.clone()
        with torch.no_grad():
            outputs = model(chunk, labels=target)
            nlls.append(outputs.loss.item())

    return {"perplexity": float(np.exp(np.mean(nlls))), "num_chunks": len(nlls)}


# ---------------------------------------------------------------------------
# Karmonic LLM Trainer
# ---------------------------------------------------------------------------

class KarmonicLLMTrainer(Trainer):
    """Custom Trainer that adds Karmonic regularization to CE loss.

    Hooks into the model's last hidden layer, mean-pools across sequence,
    passes through GradientScale(0.1) -> FourierTorusHead -> KarmonicFilterLoss.
    """

    def __init__(self, lambda_karmonic=0.05, grad_scale=0.1, hidden_dim=896,
                 torus_dim=2, n_modes=6, grid_size=12, **kwargs):
        super().__init__(**kwargs)
        self.lambda_karmonic = lambda_karmonic
        self.grad_scale = grad_scale
        self._captured_hidden = None

        # Karmonic head and loss — move to model device after init
        self.torus_head = FourierTorusHeadLLM(
            input_dim=hidden_dim, hidden_dim=128,
            torus_dim=torus_dim, n_modes=n_modes,
        )
        self.karmonic_loss_fn = KarmonicFilterLoss(
            torus_dim=torus_dim, n_modes=n_modes, grid_size=grid_size,
        )
        self._hook_handle = None
        self._device_set = False

    def _ensure_device(self):
        if not self._device_set:
            device = self.model.device
            self.torus_head = self.torus_head.to(device)
            self.karmonic_loss_fn = self.karmonic_loss_fn.to(device)
            self._device_set = True

    def _hook_fn(self, module, input, output):
        """Capture last hidden states before the LM head."""
        # output is BaseModelOutputWithPast; hidden_states is output[0]
        if isinstance(output, tuple):
            self._captured_hidden = output[0]
        else:
            self._captured_hidden = output.last_hidden_state

    def _register_hook(self):
        if self._hook_handle is not None:
            return
        # For PEFT models, the base model is wrapped
        base = self.model
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model"):
            base = base.model
        # Access the transformer's final layer norm (Qwen2 uses model.norm)
        if hasattr(base, "model") and hasattr(base.model, "norm"):
            target = base.model.norm
        elif hasattr(base, "norm"):
            target = base.norm
        elif hasattr(base, "transformer") and hasattr(base.transformer, "ln_f"):
            target = base.transformer.ln_f
        else:
            # Fallback: try to find final norm layer
            target = None
            for name, mod in base.named_modules():
                if "norm" in name.lower() and isinstance(mod, (nn.LayerNorm, nn.RMSNorm)):
                    target = mod
            if target is None:
                raise RuntimeError("Could not find final norm layer for hook")

        self._hook_handle = target.register_forward_hook(self._hook_fn)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._ensure_device()
        self._register_hook()
        self._captured_hidden = None

        # Standard forward + CE loss
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Karmonic path
        karmonic_loss = torch.tensor(0.0, device=ce_loss.device)
        if self._captured_hidden is not None and self.lambda_karmonic > 0:
            hidden = self._captured_hidden  # (B, seq_len, hidden_dim)

            # Mean pool across sequence (ignore padding via attention_mask)
            if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)

            # Gradient scale: encoder gets 10% of karmonic gradients
            pooled_scaled = gradient_scale(pooled, self.grad_scale)

            # Project to torus
            angles, fourier = self.torus_head(pooled_scaled)
            karmonic_loss = self.karmonic_loss_fn(angles, fourier)

        total_loss = ce_loss + self.lambda_karmonic * karmonic_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(use_lora=True, device_map="auto"):
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(tokenizer, max_length=512, max_samples=5000):
    print("Loading OpenAssistant oasst1...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize(example):
        text = f"User: {example['text']}\nAssistant:"
        result = tokenizer(text, truncation=True, max_length=max_length,
                           padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    return dataset.map(tokenize, remove_columns=dataset.column_names)


# ---------------------------------------------------------------------------
# Run conditions
# ---------------------------------------------------------------------------

def run_condition(condition: str, output_dir: str, eval_samples: int = None,
                  train_samples: int = 5000, lambda_karmonic: float = 0.05,
                  grad_scale_val: float = 0.1):
    """Run a single experimental condition."""
    out = Path(output_dir) / condition
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    t0 = time.time()
    use_karmonic = "karmonic" in condition
    use_lora = condition != "baseline"
    use_tlb = "tlb" in condition

    # Load model
    model, tokenizer = load_model(use_lora=use_lora)
    device = next(model.parameters()).device

    # Train if not baseline
    if use_lora:
        train_dataset = prepare_dataset(tokenizer, max_samples=train_samples)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=str(out / "checkpoints"),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )

        if use_karmonic:
            trainer = KarmonicLLMTrainer(
                lambda_karmonic=lambda_karmonic,
                grad_scale=grad_scale_val,
                hidden_dim=model.config.hidden_size,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )

        print("\nTraining...")
        trainer.train()
        trainer.save_model(str(out / "model"))

    train_time = time.time() - t0

    # Evaluate
    print("\nEvaluating...")
    logits_processor = ToroidalLogitProcessor() if use_tlb else None

    mc1 = eval_truthfulqa_mc1(model, tokenizer, logits_processor, max_samples=eval_samples)
    mc2 = eval_truthfulqa_mc2(model, tokenizer, logits_processor, max_samples=eval_samples)
    ppl = eval_perplexity(model, tokenizer, max_samples=50)

    eval_time = time.time() - t0 - train_time

    results = {
        "condition": condition,
        "model": MODEL_NAME,
        "lambda_karmonic": lambda_karmonic if use_karmonic else 0.0,
        "grad_scale": grad_scale_val if use_karmonic else 0.0,
        "use_tlb": use_tlb,
        "train_samples": train_samples if use_lora else 0,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        **mc1, **mc2, **ppl,
    }

    with open(out / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for {condition}:")
    print(f"  MC1: {mc1['mc1_accuracy']:.4f}")
    print(f"  MC2: {mc2['mc2_score']:.4f}")
    print(f"  PPL: {ppl['perplexity']:.2f}")
    print(f"  Train: {train_time:.0f}s, Eval: {eval_time:.0f}s")

    # Free GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    parser = argparse.ArgumentParser(description="Karmonic-Constrained LLM Fine-Tuning")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["baseline", "lora_only", "lora_karmonic",
                                 "lora_karmonic_tlb", "all"])
    parser.add_argument("--output_dir", type=str, default="results/karmonic_llm")
    parser.add_argument("--eval_samples", type=int, default=None,
                        help="Limit eval samples (None=full)")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--lambda_karmonic", type=float, default=0.05)
    parser.add_argument("--grad_scale", type=float, default=0.1)

    args = parser.parse_args()

    conditions = (
        ["baseline", "lora_only", "lora_karmonic", "lora_karmonic_tlb"]
        if args.condition == "all"
        else [args.condition]
    )

    all_results = []
    for cond in conditions:
        r = run_condition(
            cond, args.output_dir, eval_samples=args.eval_samples,
            train_samples=args.train_samples,
            lambda_karmonic=args.lambda_karmonic,
            grad_scale_val=args.grad_scale,
        )
        all_results.append(r)

    # Print summary table
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"{'Condition':<22} {'MC1':>8} {'MC2':>8} {'PPL':>10} {'Train(s)':>10}")
    print(f"{'-'*72}")
    for r in all_results:
        print(f"{r['condition']:<22} {r['mc1_accuracy']:>8.4f} {r['mc2_score']:>8.4f} "
              f"{r['perplexity']:>10.2f} {r['train_time_s']:>10.1f}")

    # Save combined
    combined = Path(args.output_dir) / "combined_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined}")


if __name__ == "__main__":
    main()
