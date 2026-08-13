"""
Genetic Algorithm for the Triangle Image Problem.

Public API
----------
genetic_algorithm(solution_class, cfg, selection_fn, xo_fn, mut_fn,
                  run_id=None, verbose=False)
    Main GA loop.  All hyperparameters and feature flags come from cfg.

load_or_run_ga(label, *, solution_class, cfg, selection_fn, xo_fn, mut_fn,
               verbose=False, reuse_existing=True, results_dir='results')
    Load a cached matching run from disk, or execute and persist.

hybrid_ga_then_sa(solution_class, ga_cfg, sa_cfg, run_id=None, verbose=False)
    Run GA (or load an existing run), then refine its best with SA.

load_or_run_hybrid(label, *, solution_class, ga_cfg, sa_cfg,
                   verbose=False, reuse_existing=True, results_dir='results')
    Load a cached matching hybrid run from disk, or execute and persist.

Supporting helpers
------------------
get_elite, record_stats, plateau_detected
save_run_log, load_all_runs, load_run_log, find_latest_run, load_latest_run
log_to_stats, build_logged_solution
plot_fitness_over_generations, plot_diversity, plot_image_evolution
find_plateau, scan_patience_grid
"""

import os
import json
import re
import time
from copy import deepcopy
from datetime import datetime
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .diversity import (
    compute_fitness_metrics,
    compute_pairwise_distance_stats,
    compute_shared_fitness,
)
from .selection import tournament_selection


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _resolve_results_dir(results_dir: str = 'results') -> str:
    """Resolve the results directory relative to the project root, not the cwd."""
    return results_dir if os.path.isabs(results_dir) else os.path.join(_PROJECT_ROOT, results_dir)


def _make_run_id(label: str) -> str:
    """Import make_run_id from whichever project import style is active."""
    try:
        from config import make_run_id
    except ModuleNotFoundError:
        from Computational_Intelligence_for_Optimization.config import make_run_id
    return make_run_id(label)


# ── Elite selection ───────────────────────────────────────────────────────────

def get_elite(population: list, n: int, maximization: bool) -> list:
    """Return the top-n individuals sorted by fitness."""
    if n == 0:
        return []
    return sorted(population, key=lambda ind: ind.fitness(), reverse=maximization)[:n]


# ── Stats recorder ────────────────────────────────────────────────────────────

def record_stats(stats: dict, population: list, track_diversity: bool = True):
    """Append one generation's metrics to stats."""
    fm = compute_fitness_metrics(population)
    stats['fitness_history'].append(fm['best'])
    stats['mean_fitness_history'].append(fm['mean'])
    stats['worst_fitness_history'].append(fm['worst'])
    stats['variance_history'].append(fm['variance'])
    stats['entropy_history'].append(fm['entropy'])
    if track_diversity:
        pd = compute_pairwise_distance_stats(population)
        stats['mean_dist_history'].append(pd['mean_dist'])
        stats['std_dist_history'].append(pd['std_dist'])
    else:
        stats['mean_dist_history'].append(float('nan'))
        stats['std_dist_history'].append(float('nan'))


# ── Plateau detection ─────────────────────────────────────────────────────────

def plateau_detected(stats: dict, patience: int, min_delta: float) -> bool:
    """Return True if best fitness improved by less than min_delta over the last patience gens."""
    h = stats['fitness_history']
    if len(h) <= patience:
        return False
    return (h[-(patience + 1)] - h[-1]) < min_delta


# ── Main GA loop ───────────────────────────────────────────────────────────────

def genetic_algorithm(
    solution_class,
    cfg: dict,
    selection_fn,
    xo_fn,
    mut_fn,
    run_id: str = None,
    verbose: bool = False,
):
    """
    Run the Genetic Algorithm.

    Parameters
    ----------
    solution_class : class
    cfg : dict
        All hyperparameters and feature flags.  Keys consumed:

        population_size      int    30
        max_generations      int    1000
        xo_prob              float  0.8
        mut_prob             float  0.05
        elitesize            int    1
        maximization         bool   False
        save_image_every_n_gens int 100

        track_diversity      bool   True   — log diversity metrics each generation
        fitness_sharing      bool   False
        sigma_share          float  0.1    — niche radius
        dynamic_schedule     bool   False  — linear param decay
        mut_prob_start       float  mut_prob
        mut_prob_end         float  mut_prob
        tournament_start     int    3
        tournament_end       int    3
        innovation_replace_prob float 0.0  — per-triangle randomisation
        zorder_swap_prob     float  0.0    — per-individual z-swap
        vertex_step          int    30
        color_step           int    30
        alpha_step           int    30
        patience             int    None   — early stopping look-back
        min_delta            float  0.05   — min improvement over patience gens

    selection_fn : callable — e.g. partial(tournament_selection, tournament_size=3)
    xo_fn        : callable — e.g. uniform_triangle_crossover
    mut_fn       : callable — e.g. triangle_mutation

    Returns
    -------
    best_individual : Solution
    stats : dict with keys fitness_history, mean_fitness_history, worst_fitness_history,
            variance_history, entropy_history, mean_dist_history, std_dist_history,
            cumulative_evals, stopped_early_at
    """
    population_size   = cfg['population_size']
    max_generations   = cfg['max_generations']
    xo_prob           = cfg['xo_prob']
    mut_prob          = cfg['mut_prob']
    elitesize         = cfg.get('elitesize', 1)
    maximization      = cfg.get('maximization', False)
    save_every        = cfg.get('save_image_every_n_gens', 100)
    track_diversity   = cfg.get('track_diversity', True)
    use_sharing       = cfg.get('fitness_sharing', False)
    sigma_share       = cfg.get('sigma_share', 0.1)
    dynamic_schedule  = cfg.get('dynamic_schedule', False)
    mut_start         = cfg.get('mut_prob_start', mut_prob)
    mut_end           = cfg.get('mut_prob_end', mut_prob)
    tourn_start       = cfg.get('tournament_start', cfg.get('tournament_size', 3))
    tourn_end         = cfg.get('tournament_end',   cfg.get('tournament_size', 3))
    innov_replace     = cfg.get('innovation_replace_prob', 0.0)
    zorder_swap       = cfg.get('zorder_swap_prob', 0.0)
    vertex_step       = cfg.get('vertex_step', 30)
    color_step        = cfg.get('color_step', 30)
    alpha_step        = cfg.get('alpha_step', 30)
    patience          = cfg.get('patience', None)
    min_delta         = cfg.get('min_delta', 0.05)

    population = [solution_class() for _ in range(population_size)]

    stats = {
        'fitness_history':       [],
        'mean_fitness_history':  [],
        'worst_fitness_history': [],
        'variance_history':      [],
        'entropy_history':       [],
        'mean_dist_history':     [],
        'std_dist_history':      [],
        'cumulative_evals':      [population_size],
        'stopped_early_at':      None,
    }

    start_time = time.time()
    run_dir = None
    if run_id is not None:
        run_dir = os.path.join(_resolve_results_dir(), run_id)
        os.makedirs(run_dir, exist_ok=True)

    for gen in range(max_generations):

        # 1. Dynamic schedule
        if dynamic_schedule:
            t            = gen / max_generations
            cur_mut_prob = mut_start + t * (mut_end - mut_start)
            cur_tourn    = max(2, round(tourn_start + t * (tourn_end - tourn_start)))
            cur_sel      = partial(tournament_selection, tournament_size=cur_tourn)
            cur_vertex   = vertex_step   # step sizes held constant; override via cfg if needed
            cur_color    = color_step
            cur_alpha    = alpha_step
        else:
            cur_mut_prob = mut_prob
            cur_sel      = selection_fn
            cur_vertex   = vertex_step
            cur_color    = color_step
            cur_alpha    = alpha_step

        # 2. Fitness sharing
        shared = compute_shared_fitness(population, sigma_share) if use_sharing else None

        # 3. Elitism
        elite      = get_elite(population, elitesize, maximization)
        new_pop    = [deepcopy(ind) for ind in elite]

        # 4. Generate offspring
        while len(new_pop) < population_size:
            p1 = cur_sel(population, maximization, fitness_override=shared)
            p2 = cur_sel(population, maximization, fitness_override=shared)
            c1, c2 = xo_fn(p1, p2, xo_prob)
            c1 = mut_fn(c1, cur_mut_prob,
                        vertex_step=int(cur_vertex), color_step=int(cur_color),
                        alpha_step=int(cur_alpha),
                        innovation_replace_prob=innov_replace,
                        zorder_swap_prob=zorder_swap)
            c2 = mut_fn(c2, cur_mut_prob,
                        vertex_step=int(cur_vertex), color_step=int(cur_color),
                        alpha_step=int(cur_alpha),
                        innovation_replace_prob=innov_replace,
                        zorder_swap_prob=zorder_swap)
            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        population = new_pop[:population_size]

        # 5. Record statistics
        record_stats(stats, population, track_diversity)
        stats['cumulative_evals'].append(stats['cumulative_evals'][-1] + population_size - elitesize)

        # 6. Verbose logging
        if verbose and (gen == 0 or (gen + 1) % 50 == 0):
            elapsed = time.time() - start_time
            dist_str = (f" | MeanDist: {stats['mean_dist_history'][-1]:.1f}"
                        if track_diversity else "")
            print(f"Gen {gen+1:4d}/{max_generations} | "
                  f"Best RMSE: {stats['fitness_history'][-1]:.4f}{dist_str} | "
                  f"Elapsed: {elapsed:.1f}s")

        # 7. Save image snapshot
        if run_dir is not None and (gen == 0 or (gen + 1) % save_every == 0):
            best_snap = get_elite(population, 1, maximization)[0]
            img = Image.fromarray(best_snap.draw_image().astype(np.uint8))
            img.save(os.path.join(run_dir, f"gen_{gen+1:04d}.png"))

        # 8. Early stopping
        if patience is not None and plateau_detected(stats, patience, min_delta):
            stats['stopped_early_at'] = gen + 1
            if verbose:
                print(f"\n[Early stop] Plateau at generation {gen+1}.")
            if run_dir is not None:
                best_snap = get_elite(population, 1, maximization)[0]
                Image.fromarray(best_snap.draw_image().astype(np.uint8)).save(
                    os.path.join(run_dir, f"gen_{gen+1:04d}_final.png")
                )
            break

    final_best = get_elite(population, 1, maximization)[0]
    runtime    = time.time() - start_time

    if run_dir is not None:
        Image.fromarray(final_best.draw_image().astype(np.uint8)).save(
            os.path.join(run_dir, f"gen_{len(stats['fitness_history']):04d}_final.png")
        )
        save_run_log(run_dir, cfg, stats, runtime, final_best.fitness(),
                     stopped_early_at=stats['stopped_early_at'],
                     best_repr=getattr(final_best, 'repr', None))

    if verbose:
        tag = f"early stop at gen {stats['stopped_early_at']}" if stats['stopped_early_at'] else "done"
        print(f"\nGA {tag}. Best RMSE: {final_best.fitness():.4f} | Time: {runtime:.1f}s")

    return final_best, stats


# ── Run logging ───────────────────────────────────────────────────────────────

def save_run_log(run_dir, cfg, stats, runtime, final_fitness,
                 stopped_early_at=None, best_repr=None):
    """Save experiment metadata and fitness/diversity histories to log.json."""
    def _round(lst):
        return [round(v, 6) if v == v else None for v in lst]  # NaN → None

    log = {
        'timestamp':        datetime.now().isoformat(),
        'algorithm':        'genetic_algorithm',
        'config':           cfg,
        'runtime_seconds':  round(runtime, 2),
        'final_fitness':    round(final_fitness, 6),
        'n_generations':    len(stats['fitness_history']),
        'stopped_early_at': stopped_early_at,
        'fitness_history':         _round(stats['fitness_history']),
        'mean_fitness_history':    _round(stats['mean_fitness_history']),
        'worst_fitness_history':   _round(stats['worst_fitness_history']),
        'variance_history':        _round(stats['variance_history']),
        'entropy_history':         _round(stats['entropy_history']),
        'mean_dist_history':       _round(stats['mean_dist_history']),
        'std_dist_history':        _round(stats['std_dist_history']),
        'cumulative_evals':        stats['cumulative_evals'],
    }
    if best_repr is not None:
        log['best_repr'] = list(best_repr)
    with open(os.path.join(run_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)


def load_all_runs(results_dir: str = 'results') -> list:
    """Load all log.json files from results/ and return a list of log dicts."""
    results_dir = _resolve_results_dir(results_dir)
    logs = []
    if not os.path.isdir(results_dir):
        return logs
    for run_name in sorted(os.listdir(results_dir)):
        log_path = os.path.join(results_dir, run_name, 'log.json')
        if os.path.isfile(log_path):
            with open(log_path) as f:
                data = json.load(f)
            data['run_name'] = run_name
            logs.append(data)
    return logs


def load_run_log(run_name: str, results_dir: str = 'results') -> dict | None:
    """Load one saved run by exact directory name or by prefix."""
    if not run_name:
        return None
    results_dir = _resolve_results_dir(results_dir)
    log_path = os.path.join(results_dir, run_name, 'log.json')
    if not os.path.isfile(log_path):
        run_name = find_latest_run(run_name, results_dir)
        if run_name is None:
            return None
        log_path = os.path.join(results_dir, run_name, 'log.json')
    with open(log_path) as f:
        data = json.load(f)
    data['run_name'] = run_name
    data['run_dir']  = os.path.join(results_dir, run_name)
    return data


def find_latest_run(prefix: str, results_dir: str = 'results') -> str | None:
    """Return the most recent run directory whose name starts with prefix."""
    results_dir = _resolve_results_dir(results_dir)
    if not os.path.isdir(results_dir):
        return None
    candidates = [
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
        and (name == prefix or name.startswith(f"{prefix}_"))
        and os.path.isfile(os.path.join(results_dir, name, 'log.json'))
    ]
    return sorted(candidates)[-1] if candidates else None


def load_latest_run(prefix: str, results_dir: str = 'results') -> dict | None:
    """Load the newest matching run for a given prefix."""
    run_name = find_latest_run(prefix, results_dir)
    return load_run_log(run_name, results_dir) if run_name else None


def log_to_stats(log: dict) -> dict:
    """Normalise a persisted log into the stats dict shape used in notebooks."""
    return {
        'fitness_history':       list(log.get('fitness_history', [])),
        'mean_fitness_history':  list(log.get('mean_fitness_history', [])),
        'worst_fitness_history': list(log.get('worst_fitness_history', [])),
        'variance_history':      list(log.get('variance_history', [])),
        'entropy_history':       list(log.get('entropy_history', [])),
        'mean_dist_history':     list(log.get('mean_dist_history', [])),
        'std_dist_history':      list(log.get('std_dist_history', [])),
        'cumulative_evals':      list(log.get('cumulative_evals', [])),
        'stopped_early_at':      log.get('stopped_early_at'),
    }


def _latest_snapshot_path(run_dir: str) -> str | None:
    if not os.path.isdir(run_dir):
        return None
    pngs = [n for n in os.listdir(run_dir) if n.endswith('.png')]
    if not pngs:
        return None
    def _key(name):
        m = re.search(r'(\d+)', name)
        return (int(m.group(1)) if m else -1, 1 if '_final' in name else 0, name)
    return os.path.join(run_dir, sorted(pngs, key=_key)[-1])


class _LoggedRunSolution:
    """Lightweight proxy for a saved run when the full Solution object is unavailable."""
    def __init__(self, fitness_value, image_path=None, repr_list=None, solution_class=None):
        self._fitness        = float(fitness_value)
        self._image_path     = image_path
        self._repr           = list(repr_list) if repr_list is not None else None
        self._solution_class = solution_class

    def fitness(self): return self._fitness

    def show(self, title=None, ax=None):
        import matplotlib.pyplot as plt
        if self._repr is not None and self._solution_class is not None:
            return self._solution_class(repr=list(self._repr)).show(title=title, ax=ax)
        if self._image_path is None or not os.path.isfile(self._image_path):
            raise FileNotFoundError("No saved image snapshot available for this logged run.")
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(4, 5))
        ax.imshow(Image.open(self._image_path).convert('RGB'))
        ax.set_title(title or f"RMSE = {self._fitness:.4f}")
        ax.axis('off')
        if standalone:
            plt.tight_layout(); plt.show()

    def __repr__(self):
        return f"LoggedRunSolution(fitness={self._fitness:.4f})"


def build_logged_solution(log: dict, solution_class=None):
    """Rebuild a saved run as the true Solution object or a display proxy."""
    repr_list = log.get('best_repr')
    if repr_list is not None and solution_class is not None:
        return solution_class(repr=list(repr_list))
    run_dir = log.get('run_dir') or os.path.join(_resolve_results_dir(), log['run_name'])
    return _LoggedRunSolution(
        fitness_value=log['final_fitness'],
        image_path=_latest_snapshot_path(run_dir),
        repr_list=repr_list,
        solution_class=solution_class,
    )


# ── load_or_run_ga ────────────────────────────────────────────────────────────

def load_or_run_ga(
    label: str,
    *,
    solution_class,
    cfg: dict,
    selection_fn,
    xo_fn,
    mut_fn,
    verbose: bool = False,
    reuse_existing: bool = True,
    results_dir: str = 'results',
):
    """
    Load the newest matching GA run from disk, or execute and persist it.

    Parameters
    ----------
    label : str — prefix for the run directory (e.g. "ga_baseline")
    solution_class, cfg, selection_fn, xo_fn, mut_fn — passed to genetic_algorithm
    verbose : bool
    reuse_existing : bool — if True and a matching run exists on disk, skip re-running
    results_dir : str — root results directory (default "results")

    Returns
    -------
    solution, stats, run_name, log, was_run : bool
    """
    results_dir = _resolve_results_dir(results_dir)
    if reuse_existing:
        log = load_latest_run(label, results_dir)
        if log is not None:
            return (
                build_logged_solution(log, solution_class),
                log_to_stats(log),
                log['run_name'],
                log,
                False,
            )

    run_name = _make_run_id(label)
    best, stats = genetic_algorithm(
        solution_class=solution_class,
        cfg=cfg,
        selection_fn=selection_fn,
        xo_fn=xo_fn,
        mut_fn=mut_fn,
        run_id=run_name,
        verbose=verbose,
    )
    log = load_run_log(run_name, results_dir)
    return best, stats, run_name, log, True


# ── Plateau analysis ─────────────────────────────────────────────────────────

def find_plateau(fitness_history: list, patience: int, min_delta: float) -> int | None:
    """Return the generation at which early stopping would have triggered."""
    for g in range(patience, len(fitness_history)):
        if (fitness_history[g - patience] - fitness_history[g]) < min_delta:
            return g + 1
    return None


def scan_patience_grid(fitness_history, patience_values, min_delta_values) -> list:
    """Evaluate a grid of (patience, min_delta) on one fitness history."""
    final_rmse = fitness_history[-1]
    total_gens = len(fitness_history)
    results = []
    for pat in patience_values:
        for delta in min_delta_values:
            stop_gen = find_plateau(fitness_history, pat, delta)
            if stop_gen is not None:
                rmse_at_stop = fitness_history[stop_gen - 1]
                gens_saved   = total_gens - stop_gen
                pct_saved    = 100 * gens_saved / total_gens
            else:
                rmse_at_stop = final_rmse
                gens_saved   = 0
                pct_saved    = 0.0
            results.append({
                'patience':    pat,
                'min_delta':   delta,
                'stop_gen':    stop_gen,
                'rmse_at_stop': round(rmse_at_stop, 4) if stop_gen else None,
                'final_rmse':  round(final_rmse, 4),
                'gens_saved':  gens_saved,
                'pct_saved':   round(pct_saved, 1),
            })
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_fitness_over_generations(fitness_histories, labels=None,
                                  title='Fitness over Generations', figsize=(10, 5)):
    """Plot fitness curves on the same axes.

    With <=3 histories: individual labelled lines.
    With >3 histories: mean line + ±1 SD shaded band (hides per-seed noise).
    Accepts numeric history lists or per-run dicts with 'fitness_history'.
    """
    import numpy as np
    processed = []
    for h in fitness_histories:
        if isinstance(h, dict):
            h = h.get('fitness_history', [])
        processed.append(h)

    fig, ax = plt.subplots(figsize=figsize)

    if len(processed) <= 3:
        labels = labels or [f'Run {i+1}' for i in range(len(processed))]
        for history, label in zip(processed, labels):
            ax.plot(history, label=label, linewidth=1.5)
    else:
        min_len = min(len(h) for h in processed if h)
        arr = np.array([h[:min_len] for h in processed if h])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0, ddof=1)
        gens = range(min_len)
        n    = len(arr)
        ax.plot(mean, linewidth=2, label=f'Mean (n={n})')
        ax.fill_between(gens, mean - std, mean + std, alpha=0.3, label='±1 SD')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Best RMSE (lower is better)')
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_diversity(stats: dict, title: str = 'Population Diversity Diagnostics'):
    """2×2 diagnostic panel: fitness spread, pairwise distance, variance, entropy."""
    gens = range(1, len(stats['fitness_history']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(gens, stats['fitness_history'],       label='Best',  color='steelblue',  lw=1.5)
    ax.plot(gens, stats['mean_fitness_history'],  label='Mean',  color='darkorange', lw=1.2, ls='--')
    ax.plot(gens, stats['worst_fitness_history'], label='Worst', color='tomato',     lw=1.0, ls=':')
    ax.set_xlabel('Generation'); ax.set_ylabel('RMSE')
    ax.set_title('Fitness: best / mean / worst')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    mean_d = np.array(stats['mean_dist_history'])
    std_d  = np.array(stats.get('std_dist_history', np.zeros_like(mean_d)))
    ax.plot(gens, mean_d, color='mediumseagreen', lw=1.5, label='Mean pairwise dist')
    ax.fill_between(gens, mean_d - std_d, mean_d + std_d, alpha=0.2, color='mediumseagreen')
    ax.set_xlabel('Generation'); ax.set_ylabel('Euclidean distance')
    ax.set_title('Genotypic diversity — mean pairwise distance ± std')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gens, stats['variance_history'], color='mediumpurple', lw=1.5)
    ax.set_xlabel('Generation'); ax.set_ylabel('Variance')
    ax.set_title('Phenotypic variance of RMSE')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(gens, stats['entropy_history'], color='goldenrod', lw=1.5)
    ax.set_xlabel('Generation'); ax.set_ylabel('Entropy (nats)')
    ax.set_title('Phenotypic entropy of RMSE')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_image_evolution(run_dir: str, figsize: tuple = (16, 4)):
    """Display saved PNG snapshots from a run in a single row."""
    snapshots = sorted(f for f in os.listdir(run_dir) if f.endswith('.png'))
    if not snapshots:
        print(f"No PNG snapshots in {run_dir}")
        return
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, fname in zip(axes, snapshots):
        img = Image.open(os.path.join(run_dir, fname))
        gen_num = fname.replace('gen_', '').replace('.png', '').replace('_final', '')
        try:
            ax.set_title(f"Gen {int(gen_num)}")
        except ValueError:
            ax.set_title(fname)
        ax.imshow(img); ax.axis('off')
    plt.suptitle(f"Image evolution — {run_dir}", fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig


# ── Hybrid GA → SA ────────────────────────────────────────────────────────────

def hybrid_ga_then_sa(
    solution_class,
    ga_cfg: dict,
    sa_cfg: dict,
    run_id: str = None,
    verbose: bool = False,
):
    """
    Run GA, then refine its best individual with SA.

    The GA does global exploration; SA polishes the result with local search.
    Typical budget split: 80 % GA evals, 20 % SA evals.

    Parameters
    ----------
    solution_class : class
    ga_cfg : dict  — full GA config (same keys as genetic_algorithm's cfg).
    sa_cfg : dict  — SA refinement config.  Use lower C than standalone SA
                     (e.g. 2–5 instead of 20) so SA does not undo GA's work.
                     Keys: C, L, H, max_iter, save_every_n_steps.
    run_id : str | None
        If set, saves GA artefacts under results/<run_id>/ga/ and SA artefacts
        under results/<run_id>/sa/, and writes a combined log.json.
    verbose : bool

    Returns
    -------
    best_sa : Solution
    info : dict
        ga_history      — best RMSE per GA generation
        sa_history      — best RMSE per SA outer cycle
        combined_history — ga_history + sa_history (single-axis plotting)
        ga_final_rmse
        sa_final_rmse
    """
    from functools import partial as _partial
    from .crossover import (
        uniform_triangle_crossover, single_point_triangle_crossover,
    )
    from .mutation import triangle_mutation
    from .selection import tournament_selection
    from ..simulated_annealing import simulated_annealing

    _XO_MAP = {
        'uniform':      uniform_triangle_crossover,
        'single_point': single_point_triangle_crossover,
    }

    run_dir    = None
    ga_run_id  = None
    sa_run_id  = None
    if run_id is not None:
        run_dir   = os.path.join(_resolve_results_dir(), run_id)
        os.makedirs(run_dir, exist_ok=True)
        ga_run_id = f"{run_id}/ga"
        sa_run_id = f"{run_id}/sa"

    # Phase 1 ─ GA
    xo_fn  = _XO_MAP.get(ga_cfg.get('crossover_type', 'uniform'), uniform_triangle_crossover)
    sel_fn = _partial(tournament_selection,
                      tournament_size=ga_cfg.get('tournament_size', 3))

    t0 = time.time()
    if verbose:
        print(f"[Hybrid] Phase 1: GA  (pop={ga_cfg.get('population_size')}  "
              f"gens={ga_cfg.get('max_generations')}) ...")

    best_ga, ga_stats = genetic_algorithm(
        solution_class=solution_class,
        cfg=ga_cfg,
        selection_fn=sel_fn,
        xo_fn=xo_fn,
        mut_fn=triangle_mutation,
        run_id=ga_run_id,
        verbose=verbose,
    )

    ga_rmse = best_ga.fitness()
    if verbose:
        print(f"[Hybrid] GA done — RMSE {ga_rmse:.4f}  ({time.time()-t0:.1f}s)")

    # Phase 2 ─ SA refinement
    if verbose:
        sa_evals = sa_cfg['L'] * sa_cfg['max_iter']
        print(f"[Hybrid] Phase 2: SA  (L={sa_cfg['L']}  "
              f"outer={sa_cfg['max_iter']}  evals={sa_evals}) ...")

    best_sa, sa_history = simulated_annealing(
        initial_solution=best_ga,
        C=sa_cfg['C'],
        L=sa_cfg['L'],
        H=sa_cfg['H'],
        maximization=False,
        max_iter=sa_cfg['max_iter'],
        verbose=verbose,
        run_id=sa_run_id,
        save_every_n_steps=sa_cfg.get('save_every_n_steps', 500),
        config=sa_cfg,
    )

    sa_rmse     = best_sa.fitness()
    total_time  = time.time() - t0
    if verbose:
        print(f"[Hybrid] SA done — RMSE {sa_rmse:.4f}  "
              f"(improvement {ga_rmse - sa_rmse:.4f})  total {total_time:.1f}s")

    # Persist combined log + final image
    if run_dir is not None:
        _save_hybrid_log(run_dir, ga_cfg, sa_cfg,
                         ga_stats['fitness_history'], sa_history,
                         ga_rmse, sa_rmse, total_time,
                         best_repr=getattr(best_sa, 'repr', None))
        Image.fromarray(best_sa.draw_image().astype(np.uint8)).save(
            os.path.join(run_dir, 'best_final.png')
        )

    return best_sa, {
        'ga_history':       ga_stats['fitness_history'],
        'sa_history':       sa_history,
        'combined_history': ga_stats['fitness_history'] + sa_history,
        'ga_final_rmse':    ga_rmse,
        'sa_final_rmse':    sa_rmse,
    }


def _save_hybrid_log(run_dir, ga_cfg, sa_cfg, ga_history, sa_history,
                     ga_rmse, sa_rmse, runtime, best_repr=None):
    from datetime import datetime as _dt
    def _r(lst): return [round(v, 6) for v in lst]
    log = {
        'timestamp':        _dt.now().isoformat(),
        'algorithm':        'hybrid_ga_then_sa',
        'ga_config':        ga_cfg,
        'sa_config':        sa_cfg,
        'runtime_seconds':  round(runtime, 2),
        'ga_final_rmse':    round(ga_rmse, 6),
        'sa_final_rmse':    round(sa_rmse, 6),
        'final_fitness':    round(sa_rmse, 6),
        'ga_history':       _r(ga_history),
        'sa_history':       _r(sa_history),
        'n_ga_generations': len(ga_history),
        'n_sa_cycles':      len(sa_history),
    }
    if best_repr is not None:
        log['best_repr'] = list(best_repr)
    with open(os.path.join(run_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)


def load_or_run_hybrid(
    label: str,
    *,
    solution_class,
    ga_cfg: dict,
    sa_cfg: dict,
    verbose: bool = False,
    reuse_existing: bool = True,
    results_dir: str = 'results',
):
    """
    Load the newest matching hybrid run from disk, or execute and persist it.

    Mirrors load_or_run_ga — same calling convention, same return shape.

    Returns
    -------
    solution, info, run_name, log, was_run : bool
    """
    results_dir = _resolve_results_dir(results_dir)
    if reuse_existing:
        run_name = find_latest_run(label, results_dir)
        if run_name is not None:
            log_path = os.path.join(results_dir, run_name, 'log.json')
            if os.path.isfile(log_path):
                with open(log_path) as f:
                    log = json.load(f)
                log['run_name'] = run_name
                log['run_dir']  = os.path.join(results_dir, run_name)

                repr_list = log.get('best_repr')
                solution  = (solution_class(repr=list(repr_list))
                             if repr_list is not None else None)

                info = {
                    'ga_history':       log.get('ga_history', []),
                    'sa_history':       log.get('sa_history', []),
                    'combined_history': log.get('ga_history', []) + log.get('sa_history', []),
                    'ga_final_rmse':    log.get('ga_final_rmse'),
                    'sa_final_rmse':    log.get('sa_final_rmse'),
                }
                return solution, info, run_name, log, False

    run_name = _make_run_id(label)
    best, info = hybrid_ga_then_sa(
        solution_class=solution_class,
        ga_cfg=ga_cfg,
        sa_cfg=sa_cfg,
        run_id=run_name,
        verbose=verbose,
    )
    log = None
    log_path = os.path.join(results_dir, run_name, 'log.json')
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log = json.load(f)
        log['run_name'] = run_name
        log['run_dir']  = os.path.join(results_dir, run_name)

    return best, info, run_name, log, True
