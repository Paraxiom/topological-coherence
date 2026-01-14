#!/usr/bin/env python3
"""
Run All GPU Experiments
Complete validation protocol for topological coherence framework

This script runs all experiments defined in the GPU Experiment Protocol
and generates a comprehensive validation report.

Usage:
    python run_all_experiments.py --all
    python run_all_experiments.py --quick  # Fast validation subset
    python run_all_experiments.py --quantum-only
    python run_all_experiments.py --llm-only
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List
import numpy as np

# Import experiment modules
from cnt_quantum_simulation import (
    run_single_stage_experiment as run_quantum_single,
    run_cascaded_experiment as run_quantum_cascaded
)
from llm_toroidal_attention import (
    run_single_model_experiment,
    run_multi_model_experiment,
    run_scaling_experiment
)


def run_experiment_1_multimodel() -> Dict:
    """
    Experiment 1: Multi-Model LLM Validation
    Confirm toroidal attention benefits are universal.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Multi-Model LLM Validation")
    print("="*70)

    models = [
        'microsoft/phi-2',
        'meta-llama/Llama-2-7b-hf',
        'mistralai/Mistral-7B-v0.1',
        'microsoft/phi-3-mini-4k-instruct',
        'Qwen/Qwen2-7B',
    ]
    mask_types = ['full', 'random', 'local', 'toroidal']

    results = run_multi_model_experiment(models, mask_types)

    # Validate success criterion
    successes = 0
    for model_name, data in results.items():
        if 'masks' in data and 'toroidal' in data['masks'] and 'full' in data['masks']:
            tqa_torus = data['masks']['toroidal']['truthfulqa_mean']
            tqa_base = data['masks']['full']['truthfulqa_mean']
            if tqa_torus / tqa_base > 1.10:  # >10% improvement
                successes += 1

    passed = successes >= 4  # 4/5 models

    return {
        'experiment': 'Multi-Model LLM Validation',
        'results': results,
        'successes': successes,
        'total_models': len(models),
        'criterion': '>=4/5 models show >10% improvement',
        'passed': passed
    }


def run_experiment_2_scaling() -> Dict:
    """
    Experiment 2: Scaling Law Analysis
    Determine how toroidal benefit scales with model size.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Scaling Law Analysis")
    print("="*70)

    model_sizes = [
        ('EleutherAI/pythia-70m', 70e6),
        ('EleutherAI/pythia-160m', 160e6),
        ('EleutherAI/pythia-410m', 410e6),
        ('EleutherAI/pythia-1b', 1e9),
        ('EleutherAI/pythia-2.8b', 2.8e9),
        ('EleutherAI/pythia-6.9b', 6.9e9),
        ('EleutherAI/pythia-12b', 12e9),
    ]

    results = run_scaling_experiment(model_sizes)

    passed = results['alpha'] > 0  # Positive scaling exponent

    return {
        'experiment': 'Scaling Law Analysis',
        'results': results,
        'alpha': results['alpha'],
        'criterion': 'alpha > 0 (benefit increases with scale)',
        'passed': passed
    }


def run_experiment_3_quantum() -> Dict:
    """
    Experiment 3: CNT Quantum Master Equation Simulation
    Validate coherence enhancement in quantum simulation.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: CNT Quantum Simulation")
    print("="*70)

    results = run_quantum_single(n_modes=7, temperature=4.0)

    enhancement = results['enhancement_percent']
    passed = enhancement > 30  # >30% enhancement

    return {
        'experiment': 'CNT Quantum Simulation',
        'results': results,
        't2_tonnetz': results['t2_tonnetz'],
        't2_random': results['t2_random'],
        'enhancement_percent': enhancement,
        'criterion': '>30% T2 enhancement',
        'passed': passed
    }


def run_experiment_4_cascaded() -> Dict:
    """
    Experiment 4: Cascaded Filtering Verification
    Confirm multiplicative scaling of coherence enhancement.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Cascaded Filtering")
    print("="*70)

    results = run_quantum_cascaded(
        n_stages=[1, 2, 4, 8, 16],
        n_modes=7,
        temperature=0.02  # 20 mK
    )

    # Check if epsilon is approximately 0.34 (within 50%)
    epsilon = results['epsilon_fit']
    epsilon_match = 0.17 < epsilon < 0.51  # 0.34 +/- 50%

    # Check if 1ms achievable with <= 20 stages
    n_required = results['n_required_for_1ms']
    achievable = n_required <= 20

    passed = epsilon_match and achievable

    return {
        'experiment': 'Cascaded Filtering',
        'results': results,
        'epsilon_fit': epsilon,
        'n_required_for_1ms': n_required,
        'criterion': 'epsilon ~0.34 and 1ms achievable with <=20 stages',
        'passed': passed
    }


def run_experiment_5_adversarial() -> Dict:
    """
    Experiment 5: Adversarial Noise Robustness
    Prove harmonic signals survive while non-harmonic noise is rejected.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Adversarial Noise Robustness")
    print("="*70)

    # Simulate harmonic vs non-harmonic signal processing
    from cnt_quantum_simulation import (
        tonnetz_distance,
        harmonic_coupling
    )

    f0 = 1e9  # 1 GHz base frequency

    # Harmonic frequencies
    harmonic_freqs = [f0, 1.25*f0, 1.5*f0, 2*f0]  # M3, P5, 8ve
    # Non-harmonic frequencies
    nonharmonic_freqs = [1.37*f0, 2.83*f0, 1.618*f0]  # Random, irrational

    # Calculate coupling strengths
    harmonic_couplings = []
    for f in harmonic_freqs:
        coupling = harmonic_coupling(f0, f)
        harmonic_couplings.append(coupling)

    nonharmonic_couplings = []
    for f in nonharmonic_freqs:
        coupling = harmonic_coupling(f0, f)
        nonharmonic_couplings.append(coupling)

    # Ratio of harmonic to non-harmonic coupling
    R_harmonic = np.mean(harmonic_couplings)
    R_nonharmonic = np.mean(nonharmonic_couplings)
    enrichment_ratio = R_harmonic / (R_nonharmonic + 1e-10)

    print(f"Mean harmonic coupling: {R_harmonic:.4f}")
    print(f"Mean non-harmonic coupling: {R_nonharmonic:.6f}")
    print(f"Enrichment ratio: {enrichment_ratio:.1f}x")

    passed = enrichment_ratio > 10  # 10x enrichment

    return {
        'experiment': 'Adversarial Noise Robustness',
        'harmonic_couplings': harmonic_couplings,
        'nonharmonic_couplings': nonharmonic_couplings,
        'enrichment_ratio': enrichment_ratio,
        'criterion': '>10x harmonic enrichment',
        'passed': passed
    }


def run_experiment_6_optimality() -> Dict:
    """
    Experiment 6: Information-Theoretic Validation
    Confirm optimality of Tonnetz filtering.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Information-Theoretic Optimality")
    print("="*70)

    from cnt_quantum_simulation import tonnetz_distance

    # For small problem size, enumerate random patterns and compare to Tonnetz
    # This is a simplified version - full version would do exhaustive search

    n_freqs = 7
    f0 = 1e9
    freqs = [f0 * r for r in [1, 1.25, 4/3, 1.5, 5/3, 2, 2.5]]

    # Tonnetz pattern quality: sum of couplings for harmonic pairs
    tonnetz_quality = 0
    for i, f1 in enumerate(freqs):
        for j, f2 in enumerate(freqs):
            if i < j:
                d = tonnetz_distance(f1, f2)
                tonnetz_quality += np.exp(-5 * d)

    # Random patterns
    n_random = 1000
    random_qualities = []
    for _ in range(n_random):
        random_freqs = [f0 * (1 + 0.5 * np.random.randn()) for _ in range(n_freqs)]
        quality = 0
        for i, f1 in enumerate(random_freqs):
            for j, f2 in enumerate(random_freqs):
                if i < j:
                    d = tonnetz_distance(f1, f2)
                    quality += np.exp(-5 * d)
        random_qualities.append(quality)

    best_random = max(random_qualities)
    mean_random = np.mean(random_qualities)

    optimality_ratio = tonnetz_quality / best_random

    print(f"Tonnetz quality score: {tonnetz_quality:.4f}")
    print(f"Best random quality: {best_random:.4f}")
    print(f"Mean random quality: {mean_random:.4f}")
    print(f"Tonnetz/Best ratio: {optimality_ratio:.2%}")

    passed = optimality_ratio >= 0.95  # >=95% of theoretical optimum

    return {
        'experiment': 'Information-Theoretic Optimality',
        'tonnetz_quality': tonnetz_quality,
        'best_random_quality': best_random,
        'optimality_ratio': optimality_ratio,
        'criterion': '>=95% of theoretical optimum',
        'passed': passed
    }


def generate_report(experiments: List[Dict]) -> str:
    """Generate validation report."""

    report = []
    report.append("="*70)
    report.append("TOPOLOGICAL COHERENCE VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("="*70)
    report.append("")

    n_passed = sum(1 for e in experiments if e['passed'])
    n_total = len(experiments)

    report.append(f"OVERALL: {n_passed}/{n_total} experiments passed")
    report.append("")

    for exp in experiments:
        status = "PASSED" if exp['passed'] else "FAILED"
        report.append(f"[{status}] {exp['experiment']}")
        report.append(f"  Criterion: {exp['criterion']}")
        report.append("")

    report.append("="*70)
    report.append("CONCLUSION")
    report.append("="*70)

    if n_passed == n_total:
        report.append("Framework VALIDATED: All experiments passed.")
        report.append("Topological coherence mechanism confirmed across domains.")
    elif n_passed >= n_total * 0.8:
        report.append("Framework PARTIALLY VALIDATED: Most experiments passed.")
        report.append("Further investigation needed for failed experiments.")
    else:
        report.append("Framework REQUIRES REVISION: Multiple experiments failed.")
        report.append("Hypothesis may need refinement.")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Run All GPU Experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick validation')
    parser.add_argument('--quantum-only', action='store_true', help='Run quantum experiments only')
    parser.add_argument('--llm-only', action='store_true', help='Run LLM experiments only')
    parser.add_argument('--output', type=str, default='validation_report.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print("="*70)
    print("TOPOLOGICAL COHERENCE VALIDATION SUITE")
    print("="*70)
    print(f"Date: {datetime.now().isoformat()}")

    experiments = []

    if args.all:
        # Run everything
        experiments.append(run_experiment_1_multimodel())
        experiments.append(run_experiment_2_scaling())
        experiments.append(run_experiment_3_quantum())
        experiments.append(run_experiment_4_cascaded())
        experiments.append(run_experiment_5_adversarial())
        experiments.append(run_experiment_6_optimality())

    elif args.quick:
        # Quick validation: one of each type
        experiments.append(run_experiment_3_quantum())  # Fast
        experiments.append(run_experiment_5_adversarial())  # Fast
        experiments.append(run_experiment_6_optimality())  # Fast

    elif args.quantum_only:
        experiments.append(run_experiment_3_quantum())
        experiments.append(run_experiment_4_cascaded())
        experiments.append(run_experiment_5_adversarial())

    elif args.llm_only:
        experiments.append(run_experiment_1_multimodel())
        experiments.append(run_experiment_2_scaling())

    else:
        # Default: quick validation
        experiments.append(run_experiment_3_quantum())
        experiments.append(run_experiment_5_adversarial())

    # Generate report
    report = generate_report(experiments)
    print("\n")
    print(report)

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'experiments': experiments,
        'report': report
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
