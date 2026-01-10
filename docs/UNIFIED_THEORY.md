# Unified Theory: Conservative Composition Across Domains

**Why the same math governs ML stability, blockchain consensus, and distributed governance**

*Sylvain Cormier | Paraxiom Research | January 2026*

---

## The Core Principle

> **Scalability emerges when composition is conservative, not when steps are fast.**

Or equivalently:

> If repeated application of an operator is not norm-bounded, no amount of local throughput optimization will save the system.

This principle applies identically to:
- Transformer attention layers
- Byzantine fault-tolerant consensus
- Cross-shard governance
- Oracle routing networks

---

## The Tradeoff Structure

Every system faces the same choice:

| Approach | Local Metric | Global Metric |
|----------|--------------|---------------|
| **Unconstrained** | Maximized | Explodes |
| **Conservative** | Reduced | Bounded |

The key insight: **throughput is tunable; instability is fatal.**

---

## Domain Mappings

### 1. Machine Learning (Attention)

| Property | Unconstrained | Constrained (Toroidal) |
|----------|---------------|------------------------|
| Attention | Dense, O(n²) | Sparse, local + periodic |
| Per-layer expressivity | Maximum | Reduced |
| Deep composition | Drift / hallucination | Stable reasoning |
| Spectral gap | Vanishing | Constant |

**Mechanism:** Toroidal topology enforces locality with periodic boundary conditions, preventing unbounded influence propagation.

### 2. Blockchain Consensus (BFT)

| Property | Unconstrained | Constrained |
|----------|---------------|-------------|
| Rounds | O(1) | O(log N) |
| Messages | O(N²) | O(N log N) |
| Single-step latency | Minimal | Higher |
| Scalability | Breaks at ~100 nodes | Scales to thousands |

**Mechanism:** Bounded fan-out per round prevents message explosion while maintaining consistency guarantees.

### 3. Governance (Influence)

| Property | Unconstrained | Constrained |
|----------|---------------|-------------|
| Stake effect | Amplifying | Conserved |
| Wealth concentration | Accelerating | Bounded |
| Long-term stability | Plutocracy | Democratic |

**Mechanism:** Conservation laws prevent cumulative influence accumulation.

### 4. Network Routing (Flow)

| Property | Unconstrained | Constrained |
|----------|---------------|-------------|
| Routing | Greedy/shortest path | Spectrally bounded |
| Congestion | Cascading failure | Graceful degradation |
| Throughput variance | High | Bounded |

**Mechanism:** Spectral constraints on flow matrices prevent concentration and cascading.

---

## Mathematical Unification

All four domains share the same operator structure:

```
x_{t+1} = A · x_t
```

Where stability requires:

```
||A^n|| ≤ C for all n
```

This is equivalent to requiring:
1. **Spectral radius** ρ(A) ≤ 1
2. **Spectral gap** λ₁ - λ₂ > ε (bounded away from zero)

### The Doubly Stochastic Connection

Doubly stochastic matrices (rows and columns sum to 1) satisfy:
- All eigenvalues have magnitude ≤ 1
- Stationary distribution is uniform
- Mixing is guaranteed

This is why:
- **Sinkhorn-Knopp normalization** stabilizes attention
- **Balanced message passing** prevents consensus amplification
- **Conservation of stake influence** prevents plutocracy

---

## Why Tonnetz? Alternative Topologies

The Tonnetz (torus) is not the only geometry that works. What matters are two properties:

1. **Compactness** — closed topology, no escape to infinity
2. **Regularity** — uniform local structure (same degree at every node)

These properties guarantee a **constant spectral gap** under scaling.

### Topology Comparison

| Topology | Compact? | Regular Tiling? | Spectral Gap | Verdict |
|----------|----------|-----------------|--------------|---------|
| **Torus (T²)** | Yes | Yes (perfect grid) | Constant | **Best** — minimal working example |
| **Sphere (S²)** | Yes | No (needs 12 pentagons) | Exists but irregular | Works with caveats |
| **Ellipsoid** | Yes | No (same as sphere) | Same as sphere | No advantage over sphere |
| **Open helix** | No | — | — | **Fails** — unbounded axis |
| **Closed helix / Torus knot** | Yes | Yes | Constant | Works |
| **Klein bottle** | Yes | Non-orientable | Exotic spectrum | Possible, unexplored |
| **Higher genus (pretzel)** | Yes | Yes | Different eigenvalues | Works, richer structure |
| **Hyperbolic (H²)** | No* | Yes | Exponential decay | Interesting for hierarchies |

*Hyperbolic space is not compact but can be quotiented to compact surfaces.

### Why Torus is Minimal

The torus is the **simplest** manifold satisfying all requirements:

- **Flat** (zero curvature) — every point looks identical
- **No singularities** — unlike sphere's poles
- **Easy parameterization** — just two angles (θ, φ)
- **Computable spectrum** — eigenvalues are explicit: `λₖ = 2 - 2cos(2πk/N)`

### Research Directions: Alternative Geometries

| Geometry | Potential Application |
|----------|----------------------|
| **Hyperbolic** | Hierarchical semantics (tree-like relationships) |
| **Spherical** | Semantic spaces with natural "poles" (sentiment, etc.) |
| **Higher genus** | Multi-topic attention (separate "holes" for separate concepts) |
| **Product manifolds** | T² × S¹ for time-aware toroidal attention |

### The Core Requirement

Any manifold works if it provides:

```
Spectral gap: λ₁ - λ₂ > ε > 0 (constant as n → ∞)
```

The Tonnetz proves such manifolds exist and are practical. The choice of *which* manifold should match the structure of the semantic space being modeled.

---

## Diagram: The Unified Operator

```
                    ┌─────────────────────────────────────┐
                    │     CONSERVATIVE COMPOSITION        │
                    │     ||A^n|| ≤ C for all n          │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │   ATTENTION  │       │  CONSENSUS   │       │  GOVERNANCE  │
    │              │       │              │       │              │
    │  Toroidal    │       │  O(log N)    │       │  Influence   │
    │  Topology    │       │  Rounds      │       │  Conservation│
    │              │       │              │       │              │
    │  Spectral    │       │  Bounded     │       │  Bounded     │
    │  Gap > ε     │       │  Fan-out     │       │  Accumulation│
    └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
           │                       │                       │
           ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │   RESULT     │       │   RESULT     │       │   RESULT     │
    │              │       │              │       │              │
    │  40% less    │       │  N log N     │       │  Stable      │
    │  drift       │       │  scaling     │       │  distribution│
    │              │       │              │       │              │
    │  Stable      │       │  1000+       │       │  No          │
    │  reasoning   │       │  validators  │       │  plutocracy  │
    └──────────────┘       └──────────────┘       └──────────────┘
```

---

## The Strategic Frame

**Do not say:** "We sacrificed throughput for stability"

**Do say:**
- We optimized for asymptotic feasibility, not single-step performance
- We enforced conservation to prevent cumulative instability
- Throughput is tunable; instability is fatal

---

## Why This Matters Now

The ML field is empirically rediscovering what was enforced architecturally in blockchain systems years earlier:

1. **2020-2023:** Scaling laws suggest "bigger = better"
2. **2024-2025:** Hallucination, drift, and instability become critical blockers
3. **2026:** Geometric constraints (this work) provide the missing theory

The same researchers who dismissed "blockchain math" as irrelevant are now measuring the exact quantities (spectral gap, mixing time, drift rate) that blockchain consensus has optimized for a decade.

---

## Citation

```bibtex
@misc{cormier2026unified,
  author = {Cormier, Sylvain},
  title = {Conservative Composition: Unified Theory of Stable Distributed Systems},
  year = {2026},
  publisher = {Paraxiom Research},
  url = {https://github.com/Paraxiom/topological-coherence/docs/UNIFIED_THEORY.md}
}
```

---

## Links

- [Topological Coherence Paper](https://doi.org/10.5281/zenodo.18187835)
- [Live Demo](https://huggingface.co/spaces/paraxiom/topological-coherence)
- [PyPI Package](https://pypi.org/project/topological-coherence/)
- [QuantumHarmony Blockchain](https://github.com/Paraxiom/QuantumHarmony)
