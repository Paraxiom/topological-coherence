//! # Topological Coherence
//!
//! Rust implementation of topological constraints for coherent inference.
//! Based on: "Topological Constraints for Coherent Language Models" (Cormier, 2026)
//!
//! ## Theory
//!
//! Hallucination is a geometry problem: unconstrained latent dynamics permit
//! arbitrary drift through latent space. Topological constraints (specifically
//! toroidal manifolds with constant spectral gap) bound this drift.
//!
//! ## Hierarchy
//!
//! ```text
//! mHC (Birkhoff) ⊂ ERLHS (Hamiltonian) ⊂ Karmonic (Toroidal + Spectral)
//! ```
//!
//! ## Key Results (from validation experiment)
//!
//! | Condition | Drift Rate | Interpretation |
//! |-----------|------------|----------------|
//! | Toroidal  | 0.006      | 40% lower than baseline |
//! | Random    | 0.167      | 28x worse (proves topology matters) |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use topological_coherence::{Tonnetz, ToroidalMask};
//!
//! // Create 12x12 Tonnetz (standard musical topology)
//! let tonnetz = Tonnetz::<12>::new();
//!
//! // Distance on torus
//! let d = tonnetz.distance((0, 0), (5, 7));
//!
//! // Attention mask with locality radius 2
//! let mask = ToroidalMask::new(64, 2.0, 1.0);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

use libm::{expf, cosf, fabsf};

// Substrate SCALE codec support
#[cfg(feature = "substrate")]
use parity_scale_codec::{Decode, Encode, MaxEncodedLen};
#[cfg(feature = "substrate")]
use scale_info::TypeInfo;

// =============================================================================
// TODO: Implementation Roadmap
// =============================================================================
//
// Phase 1: Core Topology - COMPLETE
// --------------------------------
// [x] Tonnetz struct with const generic grid size
// [x] Toroidal distance calculation
// [x] Basic attention mask generation
// [x] Spectral gap computation
// [x] Property tests (symmetry, triangle inequality)
//
// Phase 2: Mask Variants - COMPLETE
// ----------------------
// [x] Hard cutoff mask (M(i,j) = 1 if d <= r, else 0)
// [x] Soft exponential decay (M(i,j) = exp(-α * d))
// [x] Hybrid mask (hard within r, soft decay outside)
// [x] Sinkhorn-Knopp projection for doubly-stochastic
//
// Phase 3: Integration Points - IN PROGRESS
// ---------------------------
// [ ] Off-chain worker trait implementation
// [ ] Pallet storage types for embeddings
// [ ] Benchmarks for on-chain verification
// [ ] SCALE codec for mask serialization
//
// Phase 4: Advanced Features
// --------------------------
// [ ] Learned toroidal projection φ_θ(e) = (σ(W₁e) mod 1, σ(W₂e) mod 1)
// [ ] Adjacency loss computation
// [ ] Multi-scale Tonnetz (different grid sizes)
// [ ] Higher-dimensional tori (T^3, T^4)
//
// References:
// - Paper: https://github.com/Paraxiom/topological-coherence
// - ERLHS: DOI 10.5281/zenodo.17928909
// - Karmonic: DOI 10.5281/zenodo.17928991
//
// =============================================================================

/// Mask type variants for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "substrate", derive(Encode, Decode, MaxEncodedLen, TypeInfo))]
pub enum MaskType {
    /// Hard cutoff: M(i,j) = 1 if d <= r, else 0
    HardCutoff,
    /// Soft exponential: M(i,j) = exp(-α * d)
    SoftExponential,
    /// Hybrid: 1 if d <= r, else exp(-α * (d - r))
    Hybrid,
}

/// Tonnetz topology on a 2D torus of size N x N.
///
/// The Tonnetz is a toroidal lattice where:
/// - Horizontal edges connect by perfect fifths
/// - Vertical edges connect by major thirds
/// - Diagonal edges connect by minor thirds
///
/// We use it as a constructive existence proof of a low-genus manifold
/// with constant spectral gap λ₁ = Θ(1) for fixed N.
#[derive(Debug, Clone, Copy)]
pub struct Tonnetz<const N: usize>;

impl<const N: usize> Tonnetz<N> {
    /// Create a new Tonnetz topology.
    pub const fn new() -> Self {
        Self
    }

    /// Grid size (N x N torus).
    pub const fn size(&self) -> usize {
        N
    }

    /// Total number of positions on the torus.
    pub const fn total_positions(&self) -> usize {
        N * N
    }

    /// Convert linear index to 2D torus coordinates.
    #[inline]
    pub const fn to_coords(index: usize) -> (usize, usize) {
        (index / N, index % N)
    }

    /// Convert 2D torus coordinates to linear index.
    #[inline]
    pub const fn to_index(row: usize, col: usize) -> usize {
        (row % N) * N + (col % N)
    }

    /// Toroidal distance between two positions.
    ///
    /// Uses L1 (Manhattan) distance with wraparound:
    /// d_T(a, b) = min(|a.0 - b.0|, N - |a.0 - b.0|) + min(|a.1 - b.1|, N - |a.1 - b.1|)
    #[inline]
    pub fn distance(a: (usize, usize), b: (usize, usize)) -> usize {
        let dx = a.0.abs_diff(b.0);
        let dy = a.1.abs_diff(b.1);

        let dx_wrap = if dx > N / 2 { N - dx } else { dx };
        let dy_wrap = if dy > N / 2 { N - dy } else { dy };

        dx_wrap + dy_wrap
    }

    /// Distance between two linear indices.
    #[inline]
    pub fn distance_linear(i: usize, j: usize) -> usize {
        Self::distance(Self::to_coords(i), Self::to_coords(j))
    }

    /// First non-trivial eigenvalue of the torus Laplacian (spectral gap).
    ///
    /// For a d-dimensional torus T^d_N:
    /// λ₁ = 2 - 2cos(2π/N) = Θ(1) for fixed N
    ///
    /// This is the key property: spectral gap remains constant regardless
    /// of how we scale the embedding dimension.
    pub fn spectral_gap() -> f32 {
        let pi = core::f32::consts::PI;
        2.0 - 2.0 * cosf(2.0 * pi / N as f32)
    }

    /// Spectral gap decay rate for non-resonant modes.
    ///
    /// Non-resonant modes decay as e^(-λ₁ * t).
    pub fn decay_rate(t: f32) -> f32 {
        expf(-Self::spectral_gap() * t)
    }
}

impl<const N: usize> Default for Tonnetz<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Toroidal attention mask generator.
///
/// Creates masks according to Eq. 17 from the paper:
/// ```text
/// M_Tonnetz(i, j) = 1                        if d_Tonnetz(i, j) ≤ r
///                   exp(-α · d_Tonnetz(i,j)) otherwise
/// ```
#[derive(Debug, Clone)]
pub struct ToroidalMask {
    /// Sequence length
    pub seq_len: usize,
    /// Locality radius (hard cutoff)
    pub radius: f32,
    /// Decay rate for soft falloff
    pub alpha: f32,
    /// Grid size for Tonnetz mapping
    pub grid_size: usize,
    /// Mask type variant
    pub mask_type: MaskType,
}

impl ToroidalMask {
    /// Create a new toroidal mask configuration (hybrid by default).
    pub fn new(seq_len: usize, radius: f32, alpha: f32) -> Self {
        Self::with_grid(seq_len, radius, alpha, 12)
    }

    /// Create with custom grid size.
    pub fn with_grid(seq_len: usize, radius: f32, alpha: f32, grid_size: usize) -> Self {
        Self {
            seq_len,
            radius,
            alpha,
            grid_size,
            mask_type: MaskType::Hybrid,
        }
    }

    /// Create hard cutoff mask: M(i,j) = 1 if d <= r, else 0
    pub fn hard_cutoff(seq_len: usize, radius: f32, grid_size: usize) -> Self {
        Self {
            seq_len,
            radius,
            alpha: 0.0,
            grid_size,
            mask_type: MaskType::HardCutoff,
        }
    }

    /// Create soft exponential mask: M(i,j) = exp(-α * d)
    pub fn soft_exponential(seq_len: usize, alpha: f32, grid_size: usize) -> Self {
        Self {
            seq_len,
            radius: 0.0,
            alpha,
            grid_size,
            mask_type: MaskType::SoftExponential,
        }
    }

    /// Compute toroidal distance between two sequence positions.
    fn toroidal_distance(&self, i: usize, j: usize) -> f32 {
        let n = self.grid_size;
        let pos_i = (i % n, (i / n) % n);
        let pos_j = (j % n, (j / n) % n);

        let dx = pos_i.0.abs_diff(pos_j.0);
        let dy = pos_i.1.abs_diff(pos_j.1);

        let dx_wrap = if dx > n / 2 { n - dx } else { dx };
        let dy_wrap = if dy > n / 2 { n - dy } else { dy };
        (dx_wrap + dy_wrap) as f32
    }

    /// Compute mask value for position pair (i, j).
    pub fn value(&self, i: usize, j: usize) -> f32 {
        let dist = self.toroidal_distance(i, j);

        match self.mask_type {
            MaskType::HardCutoff => {
                if dist <= self.radius { 1.0 } else { 0.0 }
            }
            MaskType::SoftExponential => {
                expf(-self.alpha * dist)
            }
            MaskType::Hybrid => {
                if dist <= self.radius {
                    1.0
                } else {
                    expf(-self.alpha * (dist - self.radius))
                }
            }
        }
    }

    /// Generate full mask matrix.
    #[cfg(feature = "std")]
    pub fn generate(&self) -> Vec<Vec<f32>> {
        (0..self.seq_len)
            .map(|i| (0..self.seq_len).map(|j| self.value(i, j)).collect())
            .collect()
    }

    /// Generate and apply Sinkhorn-Knopp to make doubly-stochastic.
    #[cfg(feature = "std")]
    pub fn generate_doubly_stochastic(&self, iterations: usize) -> Vec<Vec<f32>> {
        let mask = self.generate();
        sinkhorn_knopp(mask, iterations)
    }
}

/// Sinkhorn-Knopp algorithm: project matrix to doubly-stochastic.
///
/// Alternately normalizes rows and columns until convergence.
/// A doubly-stochastic matrix has all rows and columns sum to 1.
///
/// This is used for mHC-style constraints (Birkhoff polytope projection).
#[cfg(feature = "std")]
pub fn sinkhorn_knopp(mut matrix: Vec<Vec<f32>>, iterations: usize) -> Vec<Vec<f32>> {
    let n = matrix.len();
    if n == 0 {
        return matrix;
    }

    for _ in 0..iterations {
        // Normalize rows
        for row in &mut matrix {
            let row_sum: f32 = row.iter().sum();
            if row_sum > 1e-10 {
                for val in row.iter_mut() {
                    *val /= row_sum;
                }
            }
        }

        // Normalize columns
        for j in 0..n {
            let col_sum: f32 = matrix.iter().map(|row| row[j]).sum();
            if col_sum > 1e-10 {
                for row in &mut matrix {
                    row[j] /= col_sum;
                }
            }
        }
    }

    matrix
}

/// Check if matrix is approximately doubly-stochastic.
#[cfg(feature = "std")]
pub fn is_doubly_stochastic(matrix: &[Vec<f32>], tolerance: f32) -> bool {
    let n = matrix.len();
    if n == 0 {
        return true;
    }

    // Check row sums
    for row in matrix {
        let sum: f32 = row.iter().sum();
        if fabsf(sum - 1.0) > tolerance {
            return false;
        }
    }

    // Check column sums
    for j in 0..n {
        let sum: f32 = (0..n).map(|i| matrix[i][j]).sum();
        if fabsf(sum - 1.0) > tolerance {
            return false;
        }
    }

    true
}

/// Drift measurement utilities.
///
/// Drift rate = fraction of transitions where d_Tonnetz(pred, target) > threshold.
pub struct DriftMeter {
    /// Distance threshold for "drift" classification
    pub threshold: usize,
    /// Number of transitions measured
    pub count: usize,
    /// Number of drifts detected
    pub drifts: usize,
}

impl DriftMeter {
    /// Create a new drift meter with given threshold.
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            count: 0,
            drifts: 0,
        }
    }

    /// Record a transition from predicted to target position.
    pub fn record<const N: usize>(&mut self, pred: usize, target: usize) {
        let dist = Tonnetz::<N>::distance_linear(pred, target);
        self.count += 1;
        if dist > self.threshold {
            self.drifts += 1;
        }
    }

    /// Current drift rate (0.0 to 1.0).
    pub fn rate(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.drifts as f32 / self.count as f32
        }
    }

    /// Reset measurements.
    pub fn reset(&mut self) {
        self.count = 0;
        self.drifts = 0;
    }
}

// =============================================================================
// Substrate Integration Types
// =============================================================================

/// Position on the Tonnetz torus (for on-chain storage).
///
/// Stores coordinates as u8 to minimize storage cost.
/// Supports grids up to 255x255.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "substrate", derive(Encode, Decode, MaxEncodedLen, TypeInfo))]
pub struct ToroidalPosition {
    /// Row coordinate (0..N)
    pub row: u8,
    /// Column coordinate (0..N)
    pub col: u8,
}

impl ToroidalPosition {
    /// Create a new position.
    pub const fn new(row: u8, col: u8) -> Self {
        Self { row, col }
    }

    /// Convert to tuple.
    pub const fn as_tuple(&self) -> (usize, usize) {
        (self.row as usize, self.col as usize)
    }

    /// Distance to another position on an NxN torus.
    pub fn distance_to<const N: usize>(&self, other: &Self) -> usize {
        Tonnetz::<N>::distance(self.as_tuple(), other.as_tuple())
    }
}

/// Configuration for coherence validation (for pallet storage).
///
/// Defines the parameters for topological coherence checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "substrate", derive(Encode, Decode, MaxEncodedLen, TypeInfo))]
pub struct CoherenceConfig {
    /// Grid size for Tonnetz topology
    pub grid_size: u8,
    /// Locality radius (scaled by 100 for fixed-point)
    pub radius_scaled: u16,
    /// Decay rate alpha (scaled by 100 for fixed-point)
    pub alpha_scaled: u16,
    /// Drift threshold for coherence violation
    pub drift_threshold: u8,
    /// Mask type to use
    pub mask_type: MaskType,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            grid_size: 12,
            radius_scaled: 200,  // 2.0
            alpha_scaled: 100,   // 1.0
            drift_threshold: 2,
            mask_type: MaskType::Hybrid,
        }
    }
}

impl CoherenceConfig {
    /// Create a new configuration.
    pub const fn new(
        grid_size: u8,
        radius: f32,
        alpha: f32,
        drift_threshold: u8,
        mask_type: MaskType,
    ) -> Self {
        Self {
            grid_size,
            radius_scaled: (radius * 100.0) as u16,
            alpha_scaled: (alpha * 100.0) as u16,
            drift_threshold,
            mask_type,
        }
    }

    /// Get radius as f32.
    pub fn radius(&self) -> f32 {
        self.radius_scaled as f32 / 100.0
    }

    /// Get alpha as f32.
    pub fn alpha(&self) -> f32 {
        self.alpha_scaled as f32 / 100.0
    }

    /// Create a ToroidalMask from this config.
    pub fn to_mask(&self, seq_len: usize) -> ToroidalMask {
        ToroidalMask {
            seq_len,
            radius: self.radius(),
            alpha: self.alpha(),
            grid_size: self.grid_size as usize,
            mask_type: self.mask_type,
        }
    }
}

/// Result of coherence validation (for on-chain reporting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "substrate", derive(Encode, Decode, MaxEncodedLen, TypeInfo))]
pub struct CoherenceResult {
    /// Number of transitions checked
    pub transitions: u32,
    /// Number of drift violations
    pub violations: u32,
    /// Whether coherence is within bounds
    pub is_coherent: bool,
}

impl CoherenceResult {
    /// Create from drift meter measurements.
    pub fn from_meter(meter: &DriftMeter, max_rate: f32) -> Self {
        let rate = meter.rate();
        Self {
            transitions: meter.count as u32,
            violations: meter.drifts as u32,
            is_coherent: rate <= max_rate,
        }
    }

    /// Drift rate (0.0 to 1.0).
    pub fn drift_rate(&self) -> f32 {
        if self.transitions == 0 {
            0.0
        } else {
            self.violations as f32 / self.transitions as f32
        }
    }
}

// =============================================================================
// Phase 4: Advanced Features
// =============================================================================

/// 3D Torus topology (T^3).
///
/// Higher-dimensional torus for richer semantic spaces.
/// Distance is L1 with wraparound in all three dimensions.
#[derive(Debug, Clone, Copy)]
pub struct Torus3D<const N: usize>;

impl<const N: usize> Torus3D<N> {
    /// Create a new 3D torus topology.
    pub const fn new() -> Self {
        Self
    }

    /// Total number of positions.
    pub const fn total_positions() -> usize {
        N * N * N
    }

    /// Convert linear index to 3D coordinates.
    #[inline]
    pub const fn to_coords(index: usize) -> (usize, usize, usize) {
        let z = index / (N * N);
        let rem = index % (N * N);
        let y = rem / N;
        let x = rem % N;
        (x, y, z)
    }

    /// Convert 3D coordinates to linear index.
    #[inline]
    pub const fn to_index(x: usize, y: usize, z: usize) -> usize {
        (z % N) * N * N + (y % N) * N + (x % N)
    }

    /// 3D toroidal distance (L1 with wraparound).
    #[inline]
    pub fn distance(a: (usize, usize, usize), b: (usize, usize, usize)) -> usize {
        let dx = a.0.abs_diff(b.0);
        let dy = a.1.abs_diff(b.1);
        let dz = a.2.abs_diff(b.2);

        let dx_wrap = if dx > N / 2 { N - dx } else { dx };
        let dy_wrap = if dy > N / 2 { N - dy } else { dy };
        let dz_wrap = if dz > N / 2 { N - dz } else { dz };

        dx_wrap + dy_wrap + dz_wrap
    }

    /// Spectral gap for 3D torus.
    pub fn spectral_gap() -> f32 {
        // For T^3, gap is same as T^2 for fixed N
        let pi = core::f32::consts::PI;
        2.0 - 2.0 * cosf(2.0 * pi / N as f32)
    }
}

impl<const N: usize> Default for Torus3D<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-scale Tonnetz configuration.
///
/// Combines multiple grid sizes for hierarchical coherence.
#[derive(Debug, Clone)]
pub struct MultiScaleTonnetz {
    /// Grid sizes (e.g., [6, 12, 24])
    pub scales: [usize; 3],
    /// Weights for each scale
    pub weights: [f32; 3],
}

impl Default for MultiScaleTonnetz {
    fn default() -> Self {
        Self {
            scales: [6, 12, 24],
            weights: [0.5, 0.3, 0.2],
        }
    }
}

impl MultiScaleTonnetz {
    /// Create with custom scales and weights.
    pub fn new(scales: [usize; 3], weights: [f32; 3]) -> Self {
        Self { scales, weights }
    }

    /// Weighted multi-scale distance.
    pub fn distance(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        let mut total = 0.0;
        for (i, &scale) in self.scales.iter().enumerate() {
            let d = Self::distance_at_scale(a, b, scale) as f32;
            total += self.weights[i] * d;
        }
        total
    }

    fn distance_at_scale(a: (usize, usize), b: (usize, usize), n: usize) -> usize {
        // Map positions to scale
        let a_scaled = (a.0 % n, a.1 % n);
        let b_scaled = (b.0 % n, b.1 % n);

        let dx = a_scaled.0.abs_diff(b_scaled.0);
        let dy = a_scaled.1.abs_diff(b_scaled.1);

        let dx_wrap = if dx > n / 2 { n - dx } else { dx };
        let dy_wrap = if dy > n / 2 { n - dy } else { dy };

        dx_wrap + dy_wrap
    }
}

/// Learned toroidal projection parameters.
///
/// Implements φ_θ(e) = (σ(W₁e) mod 1, σ(W₂e) mod 1)
/// where σ is sigmoid and e is an embedding vector.
///
/// Note: Only available with `std` feature due to Vec usage.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct LearnedProjection {
    /// Input dimension
    pub input_dim: usize,
    /// Grid size for output
    pub grid_size: usize,
    /// Weight matrix W1 (flattened, grid_size elements)
    pub w1: Vec<f32>,
    /// Weight matrix W2 (flattened, grid_size elements)
    pub w2: Vec<f32>,
}

#[cfg(feature = "std")]
impl LearnedProjection {
    /// Create a new projection with random initialization.
    pub fn new(input_dim: usize, grid_size: usize) -> Self {
        // Initialize with small random values (placeholder - use proper RNG in practice)
        let scale = 1.0 / (input_dim as f32).sqrt();
        let w1 = (0..input_dim).map(|i| ((i * 7) % 100) as f32 * scale / 100.0 - scale / 2.0).collect();
        let w2 = (0..input_dim).map(|i| ((i * 13) % 100) as f32 * scale / 100.0 - scale / 2.0).collect();
        Self { input_dim, grid_size, w1, w2 }
    }

    /// Sigmoid function.
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + expf(-x))
    }

    /// Project embedding to torus position.
    ///
    /// φ_θ(e) = (σ(W₁·e) mod 1, σ(W₂·e) mod 1) * grid_size
    pub fn project(&self, embedding: &[f32]) -> (usize, usize) {
        assert_eq!(embedding.len(), self.input_dim);

        // Compute W1 · e
        let dot1: f32 = self.w1.iter().zip(embedding.iter()).map(|(w, e)| w * e).sum();
        let x = Self::sigmoid(dot1);

        // Compute W2 · e
        let dot2: f32 = self.w2.iter().zip(embedding.iter()).map(|(w, e)| w * e).sum();
        let y = Self::sigmoid(dot2);

        // Map to grid
        let row = ((x * self.grid_size as f32) as usize) % self.grid_size;
        let col = ((y * self.grid_size as f32) as usize) % self.grid_size;

        (row, col)
    }
}

/// Adjacency loss computation.
///
/// L_topo = E[(a,b)~co-occur][d_T(φ(a), φ(b))] - λ · E[(a,c)~random][d_T(φ(a), φ(c))]
///
/// Co-occurring pairs should be close; random pairs should be far.
#[derive(Debug, Clone)]
pub struct AdjacencyLoss<const N: usize> {
    /// Regularization weight for negative samples
    pub lambda: f32,
    /// Accumulated positive pair distances
    positive_sum: f32,
    positive_count: usize,
    /// Accumulated negative pair distances
    negative_sum: f32,
    negative_count: usize,
}

impl<const N: usize> AdjacencyLoss<N> {
    /// Create a new adjacency loss tracker.
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda,
            positive_sum: 0.0,
            positive_count: 0,
            negative_sum: 0.0,
            negative_count: 0,
        }
    }

    /// Record a positive (co-occurring) pair.
    pub fn record_positive(&mut self, a: (usize, usize), b: (usize, usize)) {
        let d = Tonnetz::<N>::distance(a, b) as f32;
        self.positive_sum += d;
        self.positive_count += 1;
    }

    /// Record a negative (random) pair.
    pub fn record_negative(&mut self, a: (usize, usize), c: (usize, usize)) {
        let d = Tonnetz::<N>::distance(a, c) as f32;
        self.negative_sum += d;
        self.negative_count += 1;
    }

    /// Compute the loss.
    ///
    /// Lower is better: positive pairs close, negative pairs far.
    pub fn loss(&self) -> f32 {
        let pos_mean = if self.positive_count > 0 {
            self.positive_sum / self.positive_count as f32
        } else {
            0.0
        };

        let neg_mean = if self.negative_count > 0 {
            self.negative_sum / self.negative_count as f32
        } else {
            0.0
        };

        pos_mean - self.lambda * neg_mean
    }

    /// Reset accumulators.
    pub fn reset(&mut self) {
        self.positive_sum = 0.0;
        self.positive_count = 0;
        self.negative_sum = 0.0;
        self.negative_count = 0;
    }
}

/// Sparse mask in CSR (Compressed Sparse Row) format.
///
/// Efficient storage for sparse attention masks.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct SparseMask {
    /// Number of rows/columns
    pub size: usize,
    /// Row pointers (size + 1 elements)
    pub row_ptr: Vec<usize>,
    /// Column indices
    pub col_idx: Vec<usize>,
    /// Non-zero values
    pub values: Vec<f32>,
}

#[cfg(feature = "std")]
impl SparseMask {
    /// Create from dense mask, keeping values above threshold.
    pub fn from_dense(dense: &[Vec<f32>], threshold: f32) -> Self {
        let size = dense.len();
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for row in dense {
            for (j, &val) in row.iter().enumerate() {
                if val > threshold {
                    col_idx.push(j);
                    values.push(val);
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self { size, row_ptr, col_idx, values }
    }

    /// Create from ToroidalMask with threshold.
    pub fn from_toroidal(mask: &ToroidalMask, threshold: f32) -> Self {
        let dense = mask.generate();
        Self::from_dense(&dense, threshold)
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio (1.0 = fully sparse, 0.0 = fully dense).
    pub fn sparsity(&self) -> f32 {
        let total = self.size * self.size;
        if total == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f32 / total as f32)
        }
    }

    /// Get value at (i, j), or 0 if not stored.
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i >= self.size {
            return 0.0;
        }

        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];

        for k in start..end {
            if self.col_idx[k] == j {
                return self.values[k];
            }
        }

        0.0
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        self.row_ptr.len() * 8 + self.col_idx.len() * 8 + self.values.len() * 4
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Distance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tonnetz_distance_self() {
        let d = Tonnetz::<12>::distance((5, 5), (5, 5));
        assert_eq!(d, 0);
    }

    #[test]
    fn test_tonnetz_distance_adjacent() {
        let d = Tonnetz::<12>::distance((0, 0), (0, 1));
        assert_eq!(d, 1);
    }

    #[test]
    fn test_tonnetz_distance_wraparound() {
        // On a 12x12 torus, (0,0) to (0,11) should wrap to distance 1
        let d = Tonnetz::<12>::distance((0, 0), (0, 11));
        assert_eq!(d, 1);
    }

    #[test]
    fn test_tonnetz_distance_diagonal_wrap() {
        // (0,0) to (11,11) wraps both dimensions
        let d = Tonnetz::<12>::distance((0, 0), (11, 11));
        assert_eq!(d, 2); // 1 + 1 after wrapping
    }

    // -------------------------------------------------------------------------
    // Property Tests: Metric Space Axioms
    // -------------------------------------------------------------------------

    #[test]
    fn test_distance_symmetry() {
        // d(a, b) = d(b, a) for all points
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..12 {
                    for l in 0..12 {
                        let d1 = Tonnetz::<12>::distance((i, j), (k, l));
                        let d2 = Tonnetz::<12>::distance((k, l), (i, j));
                        assert_eq!(d1, d2, "Symmetry violated at ({},{}) <-> ({},{})", i, j, k, l);
                    }
                }
            }
        }
    }

    #[test]
    fn test_distance_identity() {
        // d(a, a) = 0 for all points
        for i in 0..12 {
            for j in 0..12 {
                let d = Tonnetz::<12>::distance((i, j), (i, j));
                assert_eq!(d, 0, "Identity violated at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_triangle_inequality() {
        // d(a, c) <= d(a, b) + d(b, c) for all points
        // Test a sample of points (full test is O(n^6))
        let points = [(0, 0), (3, 5), (7, 2), (11, 11), (6, 6), (1, 10)];
        for &a in &points {
            for &b in &points {
                for &c in &points {
                    let d_ac = Tonnetz::<12>::distance(a, c);
                    let d_ab = Tonnetz::<12>::distance(a, b);
                    let d_bc = Tonnetz::<12>::distance(b, c);
                    assert!(
                        d_ac <= d_ab + d_bc,
                        "Triangle inequality violated: d({:?},{:?})={} > d({:?},{:?})={} + d({:?},{:?})={}",
                        a, c, d_ac, a, b, d_ab, b, c, d_bc
                    );
                }
            }
        }
    }

    #[test]
    fn test_distance_non_negative() {
        // d(a, b) >= 0 for all points
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..12 {
                    for l in 0..12 {
                        let d = Tonnetz::<12>::distance((i, j), (k, l));
                        // usize is always >= 0, but verify the computation doesn't overflow
                        assert!(d <= 12, "Distance too large at ({},{}) <-> ({},{}): {}", i, j, k, l, d);
                    }
                }
            }
        }
    }

    #[test]
    fn test_max_distance_bounded() {
        // On a 12x12 torus, max L1 distance is 6+6=12 (half grid in each dim)
        let mut max_dist = 0;
        for i in 0..12 {
            for j in 0..12 {
                let d = Tonnetz::<12>::distance((0, 0), (i, j));
                if d > max_dist {
                    max_dist = d;
                }
            }
        }
        assert_eq!(max_dist, 12, "Max distance should be 12 (6+6)");
    }

    // -------------------------------------------------------------------------
    // Spectral Gap Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_spectral_gap_positive() {
        let gap = Tonnetz::<12>::spectral_gap();
        assert!(gap > 0.0);
        assert!(gap < 1.0); // For N=12, gap ≈ 0.268
    }

    #[test]
    fn test_spectral_gap_scales_with_n() {
        // Smaller N -> larger gap (more connected)
        let gap_6 = Tonnetz::<6>::spectral_gap();
        let gap_12 = Tonnetz::<12>::spectral_gap();
        let gap_24 = Tonnetz::<24>::spectral_gap();
        assert!(gap_6 > gap_12, "Gap should decrease with N");
        assert!(gap_12 > gap_24, "Gap should decrease with N");
    }

    // -------------------------------------------------------------------------
    // Mask Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_toroidal_mask_self() {
        let mask = ToroidalMask::new(64, 2.0, 1.0);
        assert_eq!(mask.value(0, 0), 1.0);
    }

    #[test]
    fn test_toroidal_mask_decay() {
        let mask = ToroidalMask::new(64, 1.0, 1.0);
        let v_near = mask.value(0, 1);
        let v_far = mask.value(0, 5);
        assert!(v_near >= v_far);
    }

    #[test]
    fn test_hard_cutoff_mask() {
        let mask = ToroidalMask::hard_cutoff(64, 2.0, 12);
        // Within radius -> 1.0
        assert_eq!(mask.value(0, 0), 1.0);
        assert_eq!(mask.value(0, 1), 1.0);
        // Outside radius -> 0.0
        assert_eq!(mask.value(0, 36), 0.0); // distance 6
    }

    #[test]
    fn test_soft_exponential_mask() {
        let mask = ToroidalMask::soft_exponential(64, 1.0, 12);
        // Self -> exp(0) = 1.0
        assert!((mask.value(0, 0) - 1.0).abs() < 1e-6);
        // Distance 1 -> exp(-1) ≈ 0.368
        let v1 = mask.value(0, 1);
        assert!((v1 - 0.368).abs() < 0.01);
    }

    #[test]
    fn test_hybrid_mask() {
        let mask = ToroidalMask::new(64, 2.0, 1.0);
        // Within radius -> 1.0
        assert_eq!(mask.value(0, 0), 1.0);
        assert_eq!(mask.value(0, 1), 1.0);
        // Just outside radius -> exp(-α*(d-r)) = exp(-1*(3-2)) = exp(-1)
        // Need to find a point at distance 3
    }

    // -------------------------------------------------------------------------
    // Sinkhorn-Knopp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sinkhorn_knopp_doubly_stochastic() {
        let mask = ToroidalMask::new(16, 2.0, 0.5);
        let ds = mask.generate_doubly_stochastic(50);
        assert!(
            is_doubly_stochastic(&ds, 0.01),
            "Sinkhorn-Knopp should produce doubly-stochastic matrix"
        );
    }

    #[test]
    fn test_sinkhorn_preserves_structure() {
        // After Sinkhorn-Knopp, nearby positions should still have higher values
        let mask = ToroidalMask::new(16, 2.0, 0.5);
        let ds = mask.generate_doubly_stochastic(50);
        // Diagonal (self-attention) should be relatively high
        let diag_avg: f32 = (0..16).map(|i| ds[i][i]).sum::<f32>() / 16.0;
        let total_avg: f32 = ds.iter().flat_map(|r| r.iter()).sum::<f32>() / 256.0;
        assert!(
            diag_avg > total_avg,
            "Diagonal should be above average after Sinkhorn"
        );
    }

    // -------------------------------------------------------------------------
    // Drift Meter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_drift_meter() {
        let mut meter = DriftMeter::new(2);
        meter.record::<12>(0, 1);  // distance 1, not drift
        meter.record::<12>(0, 6);  // distance 6, drift
        meter.record::<12>(0, 0);  // distance 0, not drift

        assert_eq!(meter.count, 3);
        assert_eq!(meter.drifts, 1);
        assert!((meter.rate() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_drift_meter_reset() {
        let mut meter = DriftMeter::new(2);
        meter.record::<12>(0, 6);
        meter.reset();
        assert_eq!(meter.count, 0);
        assert_eq!(meter.drifts, 0);
        assert_eq!(meter.rate(), 0.0);
    }

    // -------------------------------------------------------------------------
    // Coordinate Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coord_conversion_roundtrip() {
        for idx in 0..144 {
            let coords = Tonnetz::<12>::to_coords(idx);
            let back = Tonnetz::<12>::to_index(coords.0, coords.1);
            assert_eq!(idx, back, "Roundtrip failed for index {}", idx);
        }
    }

    // -------------------------------------------------------------------------
    // Substrate Integration Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_toroidal_position() {
        let pos = ToroidalPosition::new(5, 7);
        assert_eq!(pos.as_tuple(), (5, 7));
    }

    #[test]
    fn test_toroidal_position_distance() {
        let a = ToroidalPosition::new(0, 0);
        let b = ToroidalPosition::new(5, 7);
        let dist = a.distance_to::<12>(&b);
        assert_eq!(dist, Tonnetz::<12>::distance((0, 0), (5, 7)));
    }

    #[test]
    fn test_coherence_config_default() {
        let config = CoherenceConfig::default();
        assert_eq!(config.grid_size, 12);
        assert!((config.radius() - 2.0).abs() < 0.01);
        assert!((config.alpha() - 1.0).abs() < 0.01);
        assert_eq!(config.drift_threshold, 2);
        assert_eq!(config.mask_type, MaskType::Hybrid);
    }

    #[test]
    fn test_coherence_config_to_mask() {
        let config = CoherenceConfig::default();
        let mask = config.to_mask(64);
        assert_eq!(mask.seq_len, 64);
        assert_eq!(mask.grid_size, 12);
        assert_eq!(mask.mask_type, MaskType::Hybrid);
    }

    #[test]
    fn test_coherence_result_from_meter() {
        let mut meter = DriftMeter::new(2);
        meter.record::<12>(0, 1);
        meter.record::<12>(0, 6);
        meter.record::<12>(0, 0);

        let result = CoherenceResult::from_meter(&meter, 0.5);
        assert_eq!(result.transitions, 3);
        assert_eq!(result.violations, 1);
        assert!(result.is_coherent); // 0.333 < 0.5

        let strict_result = CoherenceResult::from_meter(&meter, 0.1);
        assert!(!strict_result.is_coherent); // 0.333 > 0.1
    }

    #[test]
    fn test_coherence_result_drift_rate() {
        let result = CoherenceResult {
            transitions: 100,
            violations: 25,
            is_coherent: true,
        };
        assert!((result.drift_rate() - 0.25).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Phase 4: Advanced Feature Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_torus3d_distance_self() {
        let d = Torus3D::<8>::distance((0, 0, 0), (0, 0, 0));
        assert_eq!(d, 0);
    }

    #[test]
    fn test_torus3d_distance_adjacent() {
        let d = Torus3D::<8>::distance((0, 0, 0), (1, 0, 0));
        assert_eq!(d, 1);
    }

    #[test]
    fn test_torus3d_distance_wraparound() {
        // On 8x8x8 torus, (0,0,0) to (7,0,0) should wrap to distance 1
        let d = Torus3D::<8>::distance((0, 0, 0), (7, 0, 0));
        assert_eq!(d, 1);
    }

    #[test]
    fn test_torus3d_max_distance() {
        // Max distance on 8x8x8 torus is 4+4+4=12
        let d = Torus3D::<8>::distance((0, 0, 0), (4, 4, 4));
        assert_eq!(d, 12);
    }

    #[test]
    fn test_torus3d_coord_roundtrip() {
        for idx in 0..512 {
            let (x, y, z) = Torus3D::<8>::to_coords(idx);
            let back = Torus3D::<8>::to_index(x, y, z);
            assert_eq!(idx, back, "Roundtrip failed for index {}", idx);
        }
    }

    #[test]
    fn test_multi_scale_tonnetz_default() {
        let ms = MultiScaleTonnetz::default();
        assert_eq!(ms.scales, [6, 12, 24]);
    }

    #[test]
    fn test_multi_scale_distance_same_point() {
        let ms = MultiScaleTonnetz::default();
        let d = ms.distance((0, 0), (0, 0));
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_multi_scale_distance_weighted() {
        let ms = MultiScaleTonnetz::new([6, 12, 24], [1.0, 0.0, 0.0]);
        // Only use 6x6 scale
        let d = ms.distance((0, 0), (3, 3));
        // On 6x6 torus, (0,0) to (3,3) = 3+3 = 6
        assert_eq!(d, 6.0);
    }

    #[test]
    fn test_learned_projection() {
        let proj = LearnedProjection::new(4, 12);
        let embedding = vec![1.0, 0.5, -0.5, 0.2];
        let (row, col) = proj.project(&embedding);
        assert!(row < 12);
        assert!(col < 12);
    }

    #[test]
    fn test_adjacency_loss_positive_pairs() {
        let mut loss = AdjacencyLoss::<12>::new(0.5);
        loss.record_positive((0, 0), (1, 1)); // distance 2
        loss.record_positive((0, 0), (0, 1)); // distance 1
        // Mean positive distance = 1.5
        let l = loss.loss();
        assert!((l - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_adjacency_loss_with_negatives() {
        let mut loss = AdjacencyLoss::<12>::new(0.5);
        loss.record_positive((0, 0), (1, 0)); // distance 1
        loss.record_negative((0, 0), (6, 6)); // distance 12
        // Loss = 1 - 0.5 * 12 = -5
        let l = loss.loss();
        assert!((l - (-5.0)).abs() < 0.001);
    }

    #[test]
    fn test_sparse_mask_from_toroidal() {
        let mask = ToroidalMask::hard_cutoff(16, 1.0, 4);
        let sparse = SparseMask::from_toroidal(&mask, 0.5);
        // Hard cutoff with radius 1 on 4x4 grid should have limited non-zeros
        assert!(sparse.nnz() < 16 * 16);
        assert!(sparse.sparsity() > 0.0);
    }

    #[test]
    fn test_sparse_mask_get() {
        let mask = ToroidalMask::hard_cutoff(16, 1.0, 4);
        let dense = mask.generate();
        let sparse = SparseMask::from_toroidal(&mask, 0.5);

        // Spot check some values
        for i in 0..16 {
            for j in 0..16 {
                let dense_val = dense[i][j];
                let sparse_val = sparse.get(i, j);
                if dense_val > 0.5 {
                    assert!((dense_val - sparse_val).abs() < 0.001);
                } else {
                    assert_eq!(sparse_val, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_sparse_mask_memory() {
        let mask = ToroidalMask::soft_exponential(64, 2.0, 12);
        let sparse = SparseMask::from_toroidal(&mask, 0.1);
        let dense_bytes = 64 * 64 * 4; // f32 = 4 bytes
        let sparse_bytes = sparse.memory_bytes();
        // Sparse should use less memory if sufficiently sparse
        if sparse.sparsity() > 0.5 {
            assert!(sparse_bytes < dense_bytes);
        }
    }
}
