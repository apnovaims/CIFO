"""
Simulated Annealing for the Triangle Image Problem.

Adapted from the P07 practice library (same interface).

The key addition for the image problem is that the SA logger tracks fitness
every L inner steps and saves PNG snapshots at configurable intervals —
matching the GA run logger so results can be plotted on a common axis for
the Challenge 2 comparison.

Parameters
----------
initial_solution : TriangleImageSolution
C : float — initial temperature (controls acceptance of worse solutions)
L : int   — inner iterations per temperature level
H : float — cooling divisor applied after each outer loop (C = C / H)
maximization : bool — False for RMSE minimization
max_iter : int — number of outer temperature cycles
verbose : bool
run_id : str | None — if set, saves log.json + PNG snapshots to results/<run_id>/
save_every_n_steps : int — save PNG snapshot every N inner steps
config : dict | None — hyperparameter dict saved into the log

Returns
-------
best_solution : TriangleImageSolution — best solution found across all iterations
fitness_history : list[float] — best fitness found so far, recorded after each outer cycle
"""

import os
import json
import time
import numpy as np
from copy import deepcopy
from random import random
from datetime import datetime

from PIL import Image


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def simulated_annealing(
    initial_solution,
    C: float,
    L: int,
    H: float,
    maximization: bool = False,
    max_iter: int = 1000,
    verbose: bool = False,
    run_id: str = None,
    save_every_n_steps: int = 100,
    config: dict = None,
):
    current = initial_solution
    current_fitness = current.fitness()

    # Track the global best separately from the current solution.
    # SA may accept worse solutions; we still want to record the best ever seen.
    best_ever = deepcopy(current)
    best_ever_fitness = current_fitness

    fitness_history = []    # best fitness after each outer cycle
    start_time = time.time()
    total_inner_steps = 0

    run_dir = None
    if run_id is not None:
        run_dir = os.path.join(_PROJECT_ROOT, "results", run_id)
        os.makedirs(run_dir, exist_ok=True)

    for outer in range(max_iter):
        for _ in range(L):
            neighbor = current.get_random_neighbor()
            neighbor_fitness = neighbor.fitness()

            better = (
                (maximization and neighbor_fitness >= current_fitness)
                or (not maximization and neighbor_fitness <= current_fitness)
            )

            if better:
                current = deepcopy(neighbor)
                current_fitness = neighbor_fitness
            else:
                p = np.exp(-abs(current_fitness - neighbor_fitness) / C)
                if random() < p:
                    current = deepcopy(neighbor)
                    current_fitness = neighbor_fitness

            # Update global best
            if (maximization and current_fitness > best_ever_fitness) or \
               (not maximization and current_fitness < best_ever_fitness):
                best_ever = deepcopy(current)
                best_ever_fitness = current_fitness

            total_inner_steps += 1

            # Save snapshot
            if run_dir is not None and total_inner_steps % save_every_n_steps == 0:
                img_array = best_ever.draw_image().astype(np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(run_dir, f"step_{total_inner_steps:06d}.png"))

        fitness_history.append(best_ever_fitness)
        C = C / H

        if verbose and (outer == 0 or (outer + 1) % 100 == 0):
            elapsed = time.time() - start_time
            print(f"Outer {outer + 1:4d}/{max_iter} | C={C:.4f} | "
                  f"Best RMSE: {best_ever_fitness:.4f} | Elapsed: {elapsed:.1f}s")

    runtime = time.time() - start_time

    if run_dir is not None:
        final_img = best_ever.draw_image().astype(np.uint8)
        Image.fromarray(final_img).save(
            os.path.join(run_dir, f"step_{total_inner_steps:06d}_final.png")
        )
        _save_sa_log(
            run_dir,
            config or {},
            fitness_history,
            runtime,
            best_ever_fitness,
            best_repr=getattr(best_ever, "repr", None),
        )

    if verbose:
        print(f"\nSA done. Best RMSE: {best_ever_fitness:.4f} | Total time: {runtime:.1f}s")

    return best_ever, fitness_history


def _save_sa_log(run_dir, config, fitness_history, runtime, final_fitness, best_repr=None):
    log = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": "simulated_annealing",
        "config": config,
        "runtime_seconds": round(runtime, 2),
        "final_fitness": round(final_fitness, 6),
        "fitness_history": [round(f, 6) for f in fitness_history],
        "n_outer_cycles": len(fitness_history),
    }
    if best_repr is not None:
        log["best_repr"] = list(best_repr)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
