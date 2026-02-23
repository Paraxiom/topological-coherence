# Karmonic LLM Experiment Plan

## Status: Phase 2 — Torus Detection + ENIGMA Synthesis

Last updated: 2026-02-23

---

## Phase 1 Results (Complete)

### What we ran
Karmonic-constrained LoRA fine-tuning on Qwen 2.5-0.5B-Instruct, evaluated on
TruthfulQA MC1/MC2 + WikiText perplexity. 4 conditions, 2 lambda values.

### Results

| Condition | λ | MC1 | MC2 | PPL | Train loss |
|-----------|------|--------|--------|-------|------------|
| baseline | — | 0.2778 | 0.4066 | 18.06 | — |
| lora_only | — | 0.2681 | 0.4088 | 17.56 | 2.054 |
| lora_karmonic | 0.05 | 0.2668 | 0.4082 | 17.58 | 5.391 |
| lora_karmonic+TLB | 0.05 | 0.2705 | 0.4088 | 17.59 | 5.391 |
| lora_karmonic | 0.01 | 0.2681 | 0.4082 | 17.60 | ~2.1 |
| lora_karmonic+TLB | 0.01 | 0.2693 | 0.4084 | 17.57 | ~2.1 |

### Key findings
1. Karmonic regularization does not harm model quality (perplexity improved)
2. At λ=0.05, Karmonic dominated CE loss (5.39 vs 2.05) — too aggressive
3. At λ=0.01, Karmonic matches vanilla LoRA exactly — too gentle
4. TLB stacking consistently adds +0.1-0.4pp MC1
5. **No MC1 improvement over baseline from Karmonic alone**

### Why it's flat
The training data (oasst1) is generic chat with NO truthfulness signal.
Karmonic shapes representations into a torus, but the torus structure
isn't encoding "truthful vs hallucinated" because the training data
doesn't distinguish them. The Karmonic constraint tells the model
WHAT SHAPE to be, but not WHICH DIRECTIONS on the torus correspond
to truthfulness.

---

## ENIGMA Paper Analysis (arXiv:2510.11278)

### What ENIGMA does
ABC (Australian Broadcasting Corporation) team. Single-loop trainer combining:
1. **GRPO** — on-policy RL with group-relative advantages (Fisher-Rao manifold)
2. **SAMI** — symmetric InfoNCE maximizing I(Y;C|X) between completions and
   constitutional principles (hypersphere contrastive)
3. **Sinkhorn OT** — entropic optimal transport regularizer on hidden states
   (Wasserstein geometry drift control)

Unified objective: L_ENIGMA = L_GRPO + λ_SAMI * L_SAMI + R_OT

### ENIGMA results
- Gemma-3-1B-IT, KAIST CoT-Collection (20k), LoRA r=16
- GPQA: +6.92pp (13.62% → 20.54%)
- TruthfulQA: +12.11pp (38.07% → 50.18%)
- These are MASSIVE improvements from a 1B model

### Comparison: ENIGMA vs Paraxiom (Karmonic)

| Aspect | ENIGMA | Paraxiom Karmonic |
|--------|--------|-------------------|
| Policy control | GRPO (Fisher-Rao) | Standard CE |
| Semantic binding | InfoNCE/SAMI (I(Y;C|X)) | **NONE** |
| Drift control | Sinkhorn OT (Wasserstein) | Karmonic filter (torus Laplacian) |
| Manifold | Agnostic (product of natural manifolds) | Prescriptive (T^2 Tonnetz) |
| Content signal | Constitutional principles via MI | None (structural only) |
| TruthfulQA gain | +12.11pp | ~0pp |

### Why ENIGMA works and Karmonic doesn't (yet)

ENIGMA has THREE geometric layers. Paraxiom has ONE (the Karmonic filter).

The critical missing piece is **SAMI** — the mutual information component that
tells the model WHICH DIRECTIONS on the manifold correspond to truthfulness.
ENIGMA's InfoNCE creates a semantic compass: "completions that follow these
principles should be close in representation space." Karmonic creates a structural
constraint: "representations should be toroidal." Without the compass, the torus
has no alignment to truth.

ENIGMA's Sinkhorn OT is *agnostic* — it just prevents catastrophic drift.
Karmonic is *specific* — it imposes toroidal structure with spectral filtering.
In principle, Karmonic is more informative. In practice, ENIGMA works better
because its OT is paired with SAMI for direction.

### The synthesis opportunity

Replace ENIGMA's generic Sinkhorn OT with Karmonic spectral filtering,
AND add a SAMI-style InfoNCE component:

```
L = L_GRPO + λ_SAMI * L_SAMI + λ_karmonic * L_karmonic
```

The argument: "ENIGMA shows geometry-aware regularization works. We show the
*relevant* geometry is specifically toroidal (proved by detecting torus structure
in hidden states), and Karmonic filtering is a more targeted regularizer than
generic OT because it preserves low-frequency class structure while attenuating
high-frequency noise."

---

## Phase 2: Prove the Torus is Already There

### Hypothesis
LLM hidden states naturally exhibit toroidal topology. If true, Karmonic
filtering amplifies natural structure (good). If false, Karmonic imposes
alien structure (bad) and ENIGMA's agnostic OT is the right call.

### Method: detect_torus_structure.py

#### Step 1: Extract hidden states
- Model: Qwen 2.5-0.5B-Instruct (same as Phase 1)
- Data: TruthfulQA validation set (817 questions)
- For each question: extract last-layer hidden states for:
  - The question itself (mean-pooled)
  - Each answer choice (mean-pooled)
- Label: truthful vs hallucinated based on MC1 labels
- Output: (N, 896) matrix of hidden state vectors

#### Step 2: Persistent homology (topological test)
- Use ripser or giotto-tda
- Compute Vietoris-Rips persistence diagrams on the hidden states
- Key metric: **Betti numbers**
  - β₀ = number of connected components (should be 1)
  - β₁ = number of independent cycles (torus has β₁ = 2)
  - β₂ = number of voids (torus has β₂ = 1)
- If β₁ ≥ 1 with significant persistence, there's circular/toroidal structure
- Compare β₁ for: all states, truthful-only, hallucinated-only

#### Step 3: Spectral analysis (algebraic test)
- Build k-NN graph from hidden states (k=10)
- Compute graph Laplacian eigenvalues
- Look for spectral gap pattern consistent with a torus:
  - Torus C_N × C_M has eigenvalues λ_{n,m} = (2-2cos(2πn/N)) + (2-2cos(2πm/M))
  - First nonzero eigenvalue has multiplicity 2 (two circles)
- Compute spectral gap λ₁ and compare to torus prediction

#### Step 4: Circular statistics (geometric test)
- Project hidden states to 2D using PCA
- Fit a torus model: project to angles θ₁, θ₂ via atan2
- Test for circular uniformity (Rayleigh test)
- Test for circular-linear correlation between angle and truthfulness

#### Step 5: Truthful vs hallucinated separation on detected structure
- If toroidal structure exists: project all states to torus coordinates
- Test: do truthful and hallucinated answers occupy different torus regions?
- Compute: toroidal distance between truthful centroid and hallucinated centroid
- If separation exists: Karmonic can amplify it. If not: need SAMI first.

### Expected outcomes
- **Best case**: β₁=2, clear spectral gap, truthful/hallucinated separate on torus
  → "The torus is already there, and truthfulness aligns with toroidal modes"
  → Karmonic is the right regularizer, just needs a content signal (SAMI)
- **Middle case**: β₁≥1, some circular structure, no truth/hallucination separation
  → "Circular structure exists but doesn't naturally encode truthfulness"
  → Need SAMI to bind truthfulness to torus directions, then Karmonic to amplify
- **Worst case**: β₁=0, no toroidal structure
  → "The torus is not naturally present; imposing it is the wrong approach"
  → Use ENIGMA's agnostic OT instead

### Dependencies
```
pip install ripser giotto-tda scikit-learn matplotlib torch transformers
```

---

## Phase 3: ENIGMA-Karmonic Hybrid (after torus detection)

### Architecture
```
Qwen 2.5-0.5B + LoRA
  |
  +-- L_GRPO: Group-relative policy optimization (on-policy RL)
  |     - 4 completions per prompt, group-relative advantages
  |     - DR-GRPO (sequence-level ratio clipping, ε=0.1)
  |     - CoT-format-only base reward (XML tags)
  |
  +-- L_SAMI: Symmetric InfoNCE (principle encoding)
  |     - Row: does completion identify its principle?
  |     - Column: does principle identify its completion?
  |     - Constitutional principles as positive/negative pairs
  |     - MI lower bound tracking
  |
  +-- L_karmonic: Karmonic spectral filter (replaces Sinkhorn OT)
        - Hook last hidden layer, mean-pool
        - GradientScale(0.1) → FourierTorusHead → KarmonicFilterLoss
        - Mode-weighted uniformity (low modes preserved, high attenuated)
        - More targeted than generic Wasserstein drift control
```

### Why this should work better than both
1. GRPO provides the policy improvement signal (what ENIGMA has, we lack)
2. SAMI provides the semantic compass (what ENIGMA has, we lack)
3. Karmonic provides BETTER regularization than Sinkhorn OT because:
   - It knows the specific geometry (torus) rather than being agnostic
   - It has spectral selectivity (preserve low modes, attenuate high)
   - It's computationally cheaper (no iterative Sinkhorn solve)
4. If the torus is already there (Phase 2), Karmonic amplifies natural structure
   rather than fighting against it

### Experimental conditions (Phase 3)

| # | Condition | Components |
|---|-----------|------------|
| 1 | GRPO only | L_GRPO (baseline, matches ENIGMA ablation) |
| 2 | GRPO + SAMI | L_GRPO + L_SAMI (matches ENIGMA without OT) |
| 3 | GRPO + SAMI + OT | L_GRPO + L_SAMI + Sinkhorn (full ENIGMA) |
| 4 | GRPO + SAMI + Karmonic | L_GRPO + L_SAMI + Karmonic (our hypothesis) |
| 5 | GRPO + Karmonic | L_GRPO + L_karmonic (no SAMI, tests Karmonic alone with RL) |

### Evaluation
- TruthfulQA MC1/MC2 (primary)
- GPQA (reasoning, secondary)
- WikiText perplexity (sanity)
- Persistent homology β₁ before/after (toroidal structure forming?)
- Spectral gap before/after

### Training config
- Model: Qwen 2.5-0.5B-Instruct or Gemma-3-1B-IT (match ENIGMA)
- Data: KAIST CoT-Collection (20k, same as ENIGMA) or truthfulness-focused
- LoRA: r=16, α=32, dropout=0.1, target q/k/v/o_proj
- GRPO: 4 completions/prompt, temp=1.0, DR-GRPO, β=0 (no KL)
- SAMI: row/col InfoNCE, λ_SAMI=0.05, warmup 50 steps
- Karmonic: λ_karmonic=0.01, grad_scale=0.1, grid_size=12, n_modes=6
- Hardware: RTX 4090 (RunPod) — ~4GB for 0.5B + LoRA

### Constitutional principles (for SAMI)
Adapt from ENIGMA's approach but using Paraxiom's domain:
- Positive: factual accuracy, source citation, uncertainty acknowledgment
- Negative (high-SI): procedural violations (skip verification, single source, etc.)
- Compute SI before training to validate principle quality

---

## Phase 4: Publication Strategy

### If torus IS naturally present (Phase 2 positive):
Paper title: "The Toroidal Geometry of LLM Representations: From Detection
to Karmonic-Guided Alignment"

Story:
1. We detect toroidal topology in LLM hidden states (persistent homology)
2. We show Karmonic spectral filtering amplifies this natural structure
3. We replace ENIGMA's generic Sinkhorn OT with targeted Karmonic regularization
4. We demonstrate equal or better TruthfulQA gains with lower compute

### If torus is NOT naturally present (Phase 2 negative):
Paper title: "Imposing Toroidal Structure on LLM Representations for
Hallucination Reduction: When Does Geometric Prescription Help?"

Story:
1. We show LLM hidden states lack natural toroidal structure
2. We demonstrate conditions under which imposing it helps vs. hurts
3. We compare prescriptive (Karmonic) vs agnostic (Sinkhorn) regularization
4. We identify the role of semantic content signals (SAMI) as necessary

---

## File Map

```
topological-coherence/experiments/
├── KARMONIC_LLM_PLAN.md          ← This file
├── train_karmonic_llm.py          ← Phase 1 (complete)
├── run_karmonic_llm.sh            ← Phase 1 RunPod launcher (complete)
├── detect_torus_structure.py      ← Phase 2 (torus detection)
├── train_enigma_karmonic.py       ← Phase 3 (ENIGMA hybrid)
├── run_enigma_karmonic.sh         ← Phase 3 RunPod launcher
└── results/
    ├── karmonic_llm/              ← Phase 1 results (λ=0.05)
    ├── karmonic_llm_lambda01/     ← Phase 1 results (λ=0.01)
    ├── torus_detection/           ← Phase 2 results
    └── enigma_karmonic/           ← Phase 3 results
```

## Key References
- ENIGMA: arXiv:2510.11278 (Seneque et al., Oct 2025)
- Karmonic Mesh: Cormier 2026 (Zenodo, Theorem 12.1)
- Toroidal Logit Bias: DOI 10.5281/zenodo.18516477
- V8b gradient scaling: jepa-torus/src/train_v8b.py
- SAMI: arXiv:2404.14313 (Fränken et al., 2024)
- GRPO: arXiv:2402.03300 (DeepSeekMath)
- Persistent homology for ML: Carlsson 2009, Ripser (Bauer 2021)

## RunPod Info
- Pod: smart_beige_tiger (RTX 4090 x1, $0.60/hr)
- SSH: `ssh root@149.36.0.77 -p 12759 -i ~/.ssh/id_ed25519`
- Alt: `ssh a0qvyfjxyvn6xz-64410ab7@ssh.runpod.io -i ~/.ssh/id_ed25519`
- Working dir: /root/karmonic_exp/
- Scripts uploaded via stdin pipe (workspace NFS has quota issues)
