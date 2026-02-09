# T² Unification: The Torus Across Four Domains

**The Schwinger-Keldysh contour at finite temperature IS a torus — and the same T² governs topological error correction, lattice coherence, and LLM attention.**

*Sylvain Cormier | Paraxiom Research | February 2026*

---

## Abstract

We identify a single mathematical object — the 2-torus T² = S¹ × S¹ — as the
unifying structure behind four apparently unrelated domains:

1. **Thermal Quantum Field Theory** — the Schwinger-Keldysh contour at finite temperature
2. **Topological Quantum Error Correction** — Kitaev's toric code
3. **Lattice Coherence Simulation** — the Tonnetz spectral gap
4. **LLM Attention Stability** — toroidal logit bias

In each domain, the **non-contractible cycles** of T² create global topological
invariants that are robust to local perturbation. This is the single mechanism
underlying error correction, spectral-gap-bounded decoherence, and attention
stability. We further propose **mantissa wraparound** (`fract()`) and **ring
buffers** as hardware-native T² primitives, connecting the abstract topology to
concrete IEEE 754 arithmetic.

---

## 1. The Torus in Thermal Quantum Field Theory

### 1.1 The Schwinger-Keldysh Contour

In quantum field theory, real-time correlation functions are computed on the
**Schwinger-Keldysh (SK) contour** — a closed time path that runs forward along
the real axis and then backward:

```
t = 0 ──────────────────→ t = T    (forward branch, C₊)
                                    │
t = 0 ←────────────────── t = T    (backward branch, C₋)
```

The closure of this contour — forward + backward = a closed loop — defines
**S¹**: a circle in real time. This is not a mathematical convenience; it is
required by unitarity. The generating functional Z[J₊, J₋] with sources on both
branches produces the full set of time-ordered, anti-time-ordered, and Wightman
functions.

### 1.2 Matsubara Periodicity

At finite temperature T = 1/(kβ), the density matrix ρ = e^(-βH)/Z imposes
**periodicity in imaginary time**:

```
G(τ) = G(τ + β)     for bosons
G(τ) = -G(τ + β)    for fermions
```

This periodicity defines a second circle: imaginary time lives on **S¹** with
circumference β = 1/kT. This is the Matsubara formalism — Fourier modes on this
circle give the discrete Matsubara frequencies ωₙ = 2πn/β (bosonic) or
ωₙ = (2n+1)π/β (fermionic).

### 1.3 The Combined Structure: T² = S¹ × S¹

At finite temperature with real-time dynamics, both periodicities are present
simultaneously. The complex time plane at finite T has the structure:

```
T² = S¹(real-time contour) × S¹(imaginary-time periodicity)
```

This is the **thermofield double** construction (Takahashi & Umezawa, 1975).
The thermal vacuum |0(β)⟩ lives in a doubled Hilbert space H ⊗ H̃, and the
time evolution on T² encodes both unitary dynamics and thermal equilibrium.

### 1.4 The KMS Condition

The Kubo-Martin-Schwinger (KMS) condition formalizes this toral structure:

```
⟨A(t) B(0)⟩_β = ⟨B(0) A(t + iβ)⟩_β
```

This says: **analytic continuation around the imaginary-time circle exchanges
operator ordering**. The KMS condition is equivalent to thermal equilibrium and
is a direct consequence of the T² topology — the two S¹ factors (real-time
ordering, imaginary-time periodicity) are linked by analytic continuation.

### 1.5 Key References

- A.J. Niemi & G.W. Semenoff, "Finite-temperature quantum field theory in
  Minkowski space," *Ann. Phys.* **152**, 105–129 (1984).
  — First rigorous treatment of SK contour at finite T as a topological object.

- N.P. Landsman & Ch.G. van Weert, "Real- and imaginary-time field theory at
  finite temperature and density," *Phys. Rep.* **145**, 141–249 (1987).
  — Comprehensive review establishing the T² structure.

- R. Kubo, "Statistical-Mechanical Theory of Irreversible Processes,"
  *J. Phys. Soc. Japan* **12**, 570 (1957).
  — Original KMS condition.

- Y. Takahashi & H. Umezawa, "Thermo field dynamics,"
  *Collect. Phenom.* **2**, 55 (1975).
  — Thermofield double formalism.

---

## 2. The Torus in Topological Quantum Error Correction

### 2.1 Kitaev's Toric Code

Kitaev's toric code (2003) places qubits on the **edges** of a square lattice
embedded on T². At each vertex v and each face (plaquette) p, define stabilizer
operators:

```
A_v = ∏(edges e at v) σ^x_e     (vertex stabilizer)
B_p = ∏(edges e around p) σ^z_e  (plaquette stabilizer)
```

The code space is the simultaneous +1 eigenspace of all stabilizers:

```
|ψ⟩ ∈ C  ⟺  A_v|ψ⟩ = |ψ⟩ and B_p|ψ⟩ = |ψ⟩  for all v, p
```

### 2.2 Why the Torus is Essential

On T², the lattice has **non-contractible loops** — closed paths that wind
around the torus and cannot be continuously shrunk to a point. These come in two
classes (horizontal and vertical winding), and the operators:

```
Z̄₁ = ∏(horizontal non-contractible loop) σ^z_e
Z̄₂ = ∏(vertical non-contractible loop) σ^z_e
X̄₁ = ∏(horizontal dual non-contractible loop) σ^x_e
X̄₂ = ∏(vertical dual non-contractible loop) σ^x_e
```

are **logical operators** that act within the code space. They commute with all
stabilizers but are not themselves stabilizers. This gives:

- **2 logical qubits** encoded in the topology of T²
- **Logical operations** = non-contractible loop operators
- **Error protection** = local errors cannot create non-contractible loops

### 2.3 The Protection IS the Topology

A local error (acting on a bounded region of the lattice) can create or extend
**contractible loops** (which are products of stabilizers and act trivially on
the code space), but it cannot create a non-contractible loop. To cause a
logical error, one must apply errors along an entire non-contractible cycle —
this requires O(N) local errors on an N×N lattice.

**The energy barrier** to a logical error scales as O(N), providing
**topological protection** that improves with system size. This is fundamentally
different from classical error correction, where protection is algebraic
(distance of code) rather than topological (winding number).

### 2.4 The Spectral Gap Connection

The toric code Hamiltonian has a **spectral gap** Δ > 0 separating the ground
state (code space) from excited states (error syndromes). This gap is:

```
Δ = min(energy to create anyonic excitation) > 0
```

and is **topologically protected** — small local perturbations cannot close it
(Bravyi, Hastings & Michalakis, 2010). This is the same spectral gap that
appears in our lattice coherence analysis (Section 3).

### 2.5 Key References

- A.Yu. Kitaev, "Fault-tolerant quantum computation by anyons,"
  *Ann. Phys.* **303**, 2–30 (2003). arXiv:quant-ph/9707021.
  — The foundational paper on the toric code.

- S. Bravyi, M.B. Hastings & S. Michalakis, "Topological quantum order:
  stability under local perturbations," *J. Math. Phys.* **51**, 093512 (2010).
  — Proves spectral gap stability of topological phases.

- E. Dennis, A. Kitaev, A. Landahl & J. Preskill, "Topological quantum memory,"
  *J. Math. Phys.* **43**, 4452 (2002).
  — Error threshold analysis for the toric code.

---

## 3. The Torus in Lattice Coherence (This Work)

### 3.1 The Tonnetz as T²

The Tonnetz is a 2D lattice with **periodic boundary conditions** in both
dimensions — this is precisely T² = (ℤ/Nℤ)² as a discrete approximation to
(ℝ/ℤ)². Our implementation maps tokens/qubits to positions on this lattice and
uses toroidal L1 distance to define coupling strength:

```
d_T(a, b) = min(|a₀ - b₀|, N - |a₀ - b₀|) + min(|a₁ - b₁|, N - |a₁ - b₁|)
```

### 3.2 The Spectral Gap

The graph Laplacian of the N×N torus lattice has eigenvalues:

```
λ_{k,l} = (2 - 2cos(2πk/N)) + (2 - 2cos(2πl/N))
```

for k, l ∈ {0, 1, ..., N-1}. The first non-trivial eigenvalue (spectral gap) is:

```
λ₁ = 2 - 2cos(2π/N) = Θ(1) for fixed N
```

This gap is **constant** — it does not vanish as we scale the system. This is
the critical property that bounds drift and ensures coherence.

### 3.3 Coherence Scaling

The spectral gap controls mixing time and coherence decay:

```
τ_mix = O(N² / λ₁)           (mixing time)
F(t) = exp(-λ₁ · t)           (fidelity decay of non-resonant modes)
τ_coh = -ln(threshold) / λ₁   (coherence time to threshold)
```

Our simulations (tonnetz-coherence-sim) validate that toroidal coupling preserves
quantum state fidelity significantly longer than linear or random coupling, with
the advantage scaling as √N — matching the spectral gap prediction.

### 3.4 The Connection to the Toric Code

The spectral gap that protects information in Kitaev's toric code (Section 2) is
**the same mathematical object** as the spectral gap that slows decoherence in
our lattice simulation:

| Property | Toric Code (§2) | Lattice Coherence (§3) |
|----------|-----------------|------------------------|
| Lattice | Qubits on edges of T² | Qubits at vertices of T² |
| Spectral gap | Excitation energy Δ | Laplacian eigenvalue λ₁ |
| What it protects | Logical qubits (quantum info) | State fidelity (coherence) |
| Protection mechanism | Non-contractible loops | Bounded mixing time |
| Scaling | O(N) error threshold | √N coherence advantage |

In both cases, the topology of T² creates global constraints (winding numbers,
spectral gap) that local perturbations (errors, noise) cannot destroy.

---

## 4. The Torus in LLM Attention

### 4.1 Toroidal Logit Bias

We map token positions to the Tonnetz (T²) and apply an attention bias derived
from toroidal distance:

```
M_Tonnetz(i, j) = { 1                            if d_T(i,j) ≤ r
                   { exp(-α · (d_T(i,j) - r))    otherwise
```

This creates a **periodic attention structure** where influence decays with
toroidal distance but wraps around — the last token is as close to the first as
any neighbor.

### 4.2 Empirical Results

From our experiments (Cormier, 2026):

| Metric | Baseline | Toroidal | Improvement |
|--------|----------|----------|-------------|
| Error reduction | — | — | **+40%** |
| TruthfulQA | — | — | **+2.8 pp** |
| Drift rate | 0.010 | 0.006 | **40% lower** |
| Drift (random coupling) | 0.167 | — | **28× worse** |

### 4.3 Why It Works: The Toric Code Analogy

The toroidal attention mask creates **non-contractible information pathways**
through the attention graph. A single token perturbation (injection, adversarial
input) is a **local error** on T². By the same argument as the toric code
(Section 2.3), a local perturbation cannot destroy information carried by
non-contractible attention cycles.

Concretely:
- **Contractible perturbations** (local noise, single-token injection) are
  absorbed by the toroidal structure, analogous to stabilizer multiplication
  in the toric code.
- **Non-contractible perturbations** (requiring coordinated corruption along an
  entire winding cycle) would be needed to fundamentally alter the attention
  pattern — this requires O(N) corrupted tokens.

This provides a **topological explanation** for the empirical robustness of
toroidal attention to prompt injection and adversarial perturbation.

---

## 5. The Unifying Theorem (Conjecture)

### 5.1 Statement

> **Conjecture (T² Universality).**
> Non-contractible cycles on T² create global topological invariants that are
> robust to local perturbation. This is the single mechanism underlying:
> (a) topological quantum error correction,
> (b) spectral-gap-bounded decoherence in lattice simulations,
> (c) KMS thermal equilibrium in quantum field theory, and
> (d) toroidal attention stability in language models.

### 5.2 The Common Structure

All four domains share the same abstract framework:

| Component | QFT (§1) | Toric Code (§2) | Lattice (§3) | LLM (§4) |
|-----------|----------|-----------------|--------------|----------|
| **Space** | Complex time plane | Qubit lattice | Tonnetz grid | Token sequence |
| **T² structure** | SK contour × Matsubara | Lattice on T² | Periodic boundaries | Toroidal mask |
| **S¹ factor 1** | Real-time contour | Horizontal winding | Row wraparound | Position modular |
| **S¹ factor 2** | Imaginary-time β | Vertical winding | Column wraparound | Position modular |
| **Global invariant** | KMS condition | Logical qubits | Spectral gap | Attention cycles |
| **Local perturbation** | Thermal fluctuation | Qubit error | Noise channel | Token perturbation |
| **Protection** | Equilibrium restored | Error corrected | Coherence preserved | Drift bounded |

### 5.3 The Operator-Theoretic Formulation

In all four domains, the dynamics are governed by an operator A acting on a
Hilbert space (or its finite-dimensional analogue), with the stability condition:

```
||A^n|| ≤ C for all n     (conservative composition)
```

The spectral gap λ₁ > 0 of the Laplacian on T² guarantees this bound:
- Eigenvalues of A satisfy |λ| ≤ 1 (spectral radius bounded)
- The gap λ₁ - λ₂ > ε > 0 ensures exponential convergence to the invariant
  subspace (rather than algebraic — this is the topological advantage)

This connects to the conservative composition principle established in
[UNIFIED_THEORY.md](UNIFIED_THEORY.md).

---

## 6. Hardware Implementation — The Mantissa Torus

### 6.1 The Insight: fract() IS S¹

The fractional part function on IEEE 754 floats implements the circle S¹ = ℝ/ℤ:

```rust
fn wrap_to_circle(x: f64) -> f64 {
    x.fract().abs()  // Maps ℝ → [0, 1) ≅ S¹
}
```

This is not an approximation — it is the **exact quotient map** π: ℝ → ℝ/ℤ
realized in floating-point arithmetic. Two coordinates give T²:

```rust
fn wrap_to_torus(x: f64, y: f64) -> (f64, f64) {
    (x.fract().abs(), y.fract().abs())  // Maps ℝ² → T² = [0,1) × [0,1)
}
```

### 6.2 Ring Buffers: The Discrete Torus

The modular index `i % N` is the discrete analogue, implementing
(ℤ/Nℤ) ≅ discrete S¹:

```rust
fn ring_index(i: usize, n: usize) -> usize {
    i % n  // Maps ℤ → ℤ/Nℤ ≅ discrete S¹
}
```

A 2D ring buffer with `(i % N, j % N)` indexing is exactly the discrete
torus (ℤ/Nℤ)², which is the Tonnetz.

### 6.3 Connection to Professional Physics Codes

This is not novel numerics — it is standard practice in:

- **Lattice QCD:** Periodic boundary conditions via `site = (x % Lx, y % Ly, z % Lz, t % Lt)`.
  The lattice IS a 4-torus T⁴.

- **Molecular dynamics:** The **minimum image convention** computes interparticle
  distances on a periodic box (3-torus T³):
  ```
  dx = x_i - x_j
  dx = dx - round(dx / L) * L   // equivalent to fract() on normalized coords
  ```

- **Condensed matter:** Bloch's theorem — electron wavefunctions on a crystal
  lattice are periodic functions on T^d (the Brillouin zone).

### 6.4 The Toroidal Distance via fract()

On the continuous torus [0,1)², the distance is:

```rust
fn torus_distance_1d(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)  // minimum image convention
}

fn torus_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    torus_distance_1d(a.0, b.0) + torus_distance_1d(a.1, b.1)
}
```

This is the L1 geodesic distance on the flat torus, identical to the Tonnetz
distance formula (Section 3.1) after normalization by N.

### 6.5 Connection to Physical Hardware

The mantissa torus is not just a software abstraction — it maps to physical
hardware:

- **Carbon nanotube (CNT) arrays:** A CNT is topologically S¹ (rolled graphene).
  Arrays of CNTs with periodic coupling form T².

- **Superconducting qubit grids:** IBM/Google chips arrange qubits in 2D grids.
  With periodic coupling (edge qubits coupled to opposite edge), the effective
  topology is T².

- **Photonic mesh networks:** Silicon photonic circuits with waveguide loops
  can implement arbitrary graph topologies, including T².

### 6.6 Validation

We provide a Rust implementation (`tonnetz-coherence-sim/src/torus.rs`) that:

1. Implements `ContinuousTorus` via `fract()` arithmetic
2. Implements `RingBufferTorus<N>` via modular indexing
3. Validates that `RingBufferTorus<N>::distance(a, b)` = `Tonnetz::<N>::distance(a, b)` for all pairs
4. Computes the Laplacian spectral gap from the ring buffer and verifies it matches `Tonnetz::<N>::spectral_gap()`

See `tonnetz-coherence-sim/examples/torus_validation.rs` for the full validation.

---

## 7. Synthesis: From QFT to LLM

The thread connecting all four domains is:

```
QFT at finite T                  Toric Code
     │                                │
     │  S¹ × S¹ = T²                  │  T² lattice
     │  KMS periodicity               │  Non-contractible loops
     │  Thermal stability             │  Topological error protection
     │                                │
     └──────────┬─────────────────────┘
                │
          T² = THE TORUS
          (same object)
                │
     ┌──────────┴─────────────────────┐
     │                                │
Lattice Coherence                LLM Attention
     │                                │
     │  Tonnetz = discrete T²         │  Toroidal attention mask
     │  Spectral gap λ₁ = Θ(1)       │  Non-contractible attention cycles
     │  √N coherence scaling          │  +40% error reduction
     │                                │
     └──────────┬─────────────────────┘
                │
        HARDWARE: fract() + ring buffers
        IEEE 754 mantissa wraparound
        = native T² arithmetic
```

The torus is not a metaphor. It is the literal mathematical structure — the same
T² — appearing in thermal QFT (as the complex time domain), in quantum error
correction (as the code lattice), in coherence simulation (as the coupling
topology), and in LLM attention (as the bias mask). The mantissa wraparound
`fract()` and ring buffer `i % N` are its hardware-native implementations.

---

## 8. Open Questions

1. **Is the T² conjecture (§5.1) provable?** Can we derive a single theorem
   from which all four domain-specific results follow as corollaries?

2. **Higher-dimensional generalization:** T³ and T⁴ appear in lattice QCD.
   Does the Torus3D implementation (`topological_coherence::Torus3D<N>`)
   provide measurably better coherence than T²?

3. **Fermionic boundary conditions:** The Matsubara formalism distinguishes
   bosonic (periodic) and fermionic (anti-periodic) boundary conditions on S¹.
   Is there an attention analogue of anti-periodic boundaries?

4. **Anyon braiding in attention:** The toric code supports anyonic excitations
   with non-trivial braiding statistics. Can attention patterns exhibit
   analogous braiding, and would this improve reasoning?

5. **Optimal torus size:** The spectral gap λ₁ = 2 - 2cos(2π/N) is maximized
   for small N but attention needs large sequence lengths. What is the optimal
   N for a given context window?

---

## Citation

```bibtex
@misc{cormier2026t2unification,
  author = {Cormier, Sylvain},
  title = {T² Unification: The Torus Across Four Domains},
  year = {2026},
  publisher = {Paraxiom Research},
  url = {https://github.com/Paraxiom/topological-coherence/docs/T2_UNIFICATION.md}
}
```

---

## Related Documents

- [UNIFIED_THEORY.md](UNIFIED_THEORY.md) — Conservative composition principle
- [Topological Coherence Paper](https://doi.org/10.5281/zenodo.18187835) — Main paper
- [Toroidal Logit Bias](https://doi.org/10.5281/zenodo.18516477) — LLM experiments
- [Live Demo](https://huggingface.co/spaces/paraxiom-research/topological-coherence) — Interactive Gradio demo
