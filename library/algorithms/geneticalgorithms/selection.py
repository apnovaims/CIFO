"""
Selection operators for the Genetic Algorithm.

Copied and adapted from the P09 practice library (identical interface).

Both operators return a deep copy of the selected individual so the population
list is never mutated in place.

Tournament selection is preferred for this project because:
- RMSE values are small floats (typically 30–120); FPS with 1/fitness can
  introduce numerical instability when fitness values are very close.
- Tournament selection pressure is easy to control via tournament_size.

fitness_override parameter (tournament_selection only):
    When fitness sharing is active, ga.py pre-computes a shared
    fitness array and passes it here so selection uses the modified values
    without touching the cached .fitness() on each Solution.  Pass None
    (default) to use the normal cached fitness.
"""

import random
from copy import deepcopy
from ...problems.solution import Solution


def fitness_proportionate_selection(population: list, maximization: bool, **kwargs):
    """
    Roulette-wheel (fitness proportionate) selection.

    For maximization: P(select i) ∝ fitness(i)
    For minimization: P(select i) ∝ 1/fitness(i)

    Parameters
    ----------
    population : list[Solution]
    maximization : bool
    **kwargs : accepted but ignored (allows uniform call-site with tournament_selection)

    Returns
    -------
    Solution — a deep copy of the selected individual.
    """
    if maximization:
        fitness_values = [ind.fitness() for ind in population]
    else:
        fitness_values = [1.0 / ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)
    r = random.uniform(0, total_fitness)
    cumulative = 0.0
    for ind, f in zip(population, fitness_values):
        cumulative += f
        if r <= cumulative:
            return deepcopy(ind)
    return deepcopy(population[-1])


def tournament_selection(
    population: list,
    maximization: bool,
    tournament_size: int = 2,
    fitness_override: list = None,
):
    """
    Tournament selection.

    Randomly samples tournament_size individuals and returns the best one.
    Higher tournament_size → more selection pressure toward the best individuals.

    Parameters
    ----------
    population : list[Solution]
    maximization : bool
    tournament_size : int  (default 2)
    fitness_override : list[float] | None
        If provided, must be the same length as population.  Selection is done
        using these values instead of calling .fitness() on each individual.
        Used by ga.py when fitness sharing is active.

    Returns
    -------
    Solution — a deep copy of the winner.
    """
    indices = random.choices(range(len(population)), k=tournament_size)

    if fitness_override is not None:
        key = lambda k: fitness_override[k]
    else:
        key = lambda k: population[k].fitness()

    winner_idx = (max if maximization else min)(indices, key=key)
    return deepcopy(population[winner_idx])
