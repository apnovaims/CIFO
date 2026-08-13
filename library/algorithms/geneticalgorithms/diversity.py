"""
Population diversity metrics for the Triangle Image GA.

Diversity collapse is the primary failure mode.  These
functions instrument every generation so premature convergence can be
detected and attributed to a specific intervention.

Three public functions:
    compute_fitness_metrics       — phenotypic statistics of the fitness distribution
    compute_pairwise_distance_stats — genotypic diversity via pairwise Euclidean distance
    compute_shared_fitness        — fitness-sharing modified fitness
"""

import numpy as np


def compute_fitness_metrics(population: list) -> dict:
    """
    Compute population-level fitness statistics.

    Parameters
    ----------
    population : list[Solution]  — all individuals, fitness already cached.

    Returns
    -------
    dict with keys: best, mean, worst, variance, entropy
    """
    fitnesses = np.array([ind.fitness() for ind in population], dtype=np.float64)
    n_bins = min(20, len(population))
    counts, _ = np.histogram(fitnesses, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    return {
        "best":     float(fitnesses.min()),
        "mean":     float(fitnesses.mean()),
        "worst":    float(fitnesses.max()),
        "variance": float(fitnesses.var()),
        "entropy":  entropy,
    }


def compute_pairwise_distance_stats(population: list) -> dict:
    """
    Mean and std of pairwise Euclidean distances across the genome.

    Uses the ||a-b||² = ||a||² + ||b||² - 2·aᵀb identity to compute all
    distances in a single matrix multiply (one BLAS call, much faster than
    an explicit double loop).

    Parameters
    ----------
    population : list[Solution]

    Returns
    -------
    dict with keys: mean_dist, std_dist
    """
    n = len(population)
    if n < 2:
        return {"mean_dist": 0.0, "std_dist": 0.0}

    g = np.array([ind.repr for ind in population], dtype=np.float32)  # (n, m)
    sq = (g ** 2).sum(axis=1)                                          # (n,)
    D_sq = sq[:, None] + sq[None, :] - 2.0 * (g @ g.T)               # (n, n)
    D = np.sqrt(np.clip(D_sq, 0.0, None))                              # (n, n)
    upper = D[np.triu_indices(n, k=1)]                                 # n*(n-1)/2
    return {"mean_dist": float(upper.mean()), "std_dist": float(upper.std())}


def compute_shared_fitness(population: list, sigma_share: float = 0.1) -> np.ndarray:
    """
    Fitness-sharing modified fitness values for minimization.

    The sharing function S(d) = max(0, 1 - d_norm/σ_share) is applied to all
    pairwise normalised distances.  The niche count σ(x) = 1 + Σ_{y≠x} S(d(x,y))
    scales each individual's fitness upward in crowded regions:

        f_S(x) = f(x) · σ(x)

    For minimisation, higher f_S means worse — so crowded individuals are
    penalised in tournament selection when fitness_override is used.

    Parameters
    ----------
    population  : list[Solution]
    sigma_share : float — niche radius in normalised-distance space (0–1).
                  Smaller values create narrower niches (more sub-populations).

    Returns
    -------
    np.ndarray of shape (n,) — shared fitness values, one per individual.
    """
    n = len(population)
    fitnesses = np.array([ind.fitness() for ind in population], dtype=np.float64)

    if n < 2:
        return fitnesses.copy()

    g = np.array([ind.repr for ind in population], dtype=np.float32)
    sq = (g ** 2).sum(axis=1)
    D_sq = sq[:, None] + sq[None, :] - 2.0 * (g @ g.T)
    D = np.sqrt(np.clip(D_sq, 0.0, None))

    max_dist = D.max()
    D_norm = D / max_dist if max_dist > 0 else D

    sharing = np.maximum(0.0, 1.0 - D_norm / sigma_share)
    np.fill_diagonal(sharing, 0.0)

    niche_counts = 1.0 + sharing.sum(axis=1)
    return fitnesses * niche_counts
