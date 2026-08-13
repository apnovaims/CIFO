"""
Utility module for multi-seed GA experiments and statistical analysis.
All top-level functions are picklable for macOS spawn multiprocessing.
Replaces seed_worker.py.
"""
import os, sys, random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── shared worker bootstrap ───────────────────────────────────────────────────

def _worker_imports():
    """Import and return common objects. Called inside each spawned process
    because macOS spawn starts a blank interpreter."""
    sys.path.insert(0, _PROJECT_DIR)
    os.chdir(_PROJECT_DIR)

    from functools import partial
    from library.problems.triangle_image import TriangleImageSolution
    from library.algorithms.geneticalgorithms.ga import (
        load_or_run_ga, load_run_log, build_logged_solution,
    )
    from library.algorithms.geneticalgorithms.selection import tournament_selection
    from library.algorithms.geneticalgorithms.crossover import (
        uniform_triangle_crossover, single_point_triangle_crossover,
    )
    from library.algorithms.geneticalgorithms.mutation import triangle_mutation
    from library.algorithms.simulated_annealing import simulated_annealing

    TriangleImageSolution.load_target(
        os.path.join(_PROJECT_DIR, 'girl_pearl_earing.png')
    )
    return (TriangleImageSolution, load_or_run_ga, load_run_log,
            build_logged_solution, tournament_selection,
            uniform_triangle_crossover, single_point_triangle_crossover,
            triangle_mutation, simulated_annealing, partial)


def _build_xo_sel(cfg, uniform_xo, sp_xo, tournament_sel, partial):
    xo  = sp_xo if cfg.get('crossover_type') == 'single_point' else uniform_xo
    sel = partial(tournament_sel, tournament_size=cfg.get('tournament_size', 3))
    return xo, sel


# ── migrated from seed_worker.py (unchanged behaviour) ───────────────────────

def run_one_seed(seed, ga_cfg, sa_cfg):
    """Run GA + SA for one seed (Section 9).
    Returns: (seed, ga_rmse, sa_rmse, ga_fitness_history, sa_fitness_history)"""
    (Tri, load_or_run_ga, load_run_log, build_logged_solution,
     tournament_sel, uniform_xo, sp_xo, mut, sa_fn, partial) = _worker_imports()

    random.seed(seed)
    np.random.seed(seed)

    xo, sel = _build_xo_sel(ga_cfg, uniform_xo, sp_xo, tournament_sel, partial)
    best_ga, stats_ga, _, _, _ = load_or_run_ga(
        f'challenge2_ga_seed{seed}',
        solution_class=Tri, cfg=ga_cfg,
        selection_fn=sel, xo_fn=xo, mut_fn=mut, verbose=False,
    )
    ga_rmse    = best_ga.fitness()
    ga_history = stats_ga['fitness_history']

    label_sa = f'challenge2_sa_seed{seed}'
    log_sa = load_run_log(label_sa) if os.path.isdir(os.path.join('results', label_sa)) else None
    if log_sa is None:
        random.seed(seed)
        np.random.seed(seed)
        best_sa, hist_sa = sa_fn(
            initial_solution=Tri(),
            C=sa_cfg['C'], L=sa_cfg['L'], H=sa_cfg['H'],
            maximization=False, max_iter=sa_cfg['max_iter'],
            run_id=label_sa,
            save_every_n_steps=sa_cfg['save_every_n_steps'],
            config=sa_cfg,
        )
    else:
        best_sa = build_logged_solution(log_sa, Tri)
        hist_sa = list(log_sa.get('fitness_history', []))

    sa_rmse = best_sa.fitness()
    print(f"  [worker] Seed {seed}: GA={ga_rmse:.4f}, SA={sa_rmse:.4f}", flush=True)
    return seed, ga_rmse, sa_rmse, ga_history, hist_sa


def run_hybrid_seed(args):
    """Worker: load a cached GA solution and run SA refinement from it (Section 10).
    args: (seed, ga_result_label, sa_cfg)
    ga_result_label — the results/ subdirectory holding the GA log, e.g. 'paired_swap_1000_seed3'
    Returns: (seed, ga_rmse, hybrid_rmse, sa_history)
    """
    seed, ga_result_label, sa_cfg = args
    (Tri, load_or_run_ga, load_run_log, build_logged_solution,
     tournament_sel, uniform_xo, sp_xo, mut, sa_fn, partial) = _worker_imports()

    log_ga = load_run_log(ga_result_label)
    ga_sol = build_logged_solution(log_ga, Tri)
    ga_rmse = float(ga_sol.fitness())

    hybrid_label = 'hybrid10_' + ga_result_label
    if not os.path.isdir(os.path.join('results', hybrid_label)):
        random.seed(seed)
        np.random.seed(seed)
        best_sa, hist_sa = sa_fn(
            initial_solution=ga_sol,
            C=sa_cfg['C'], L=sa_cfg['L'], H=sa_cfg['H'],
            maximization=False, max_iter=sa_cfg['max_iter'],
            run_id=hybrid_label,
            save_every_n_steps=sa_cfg.get('save_every_n_steps', 1000),
            config=sa_cfg,
        )
    else:
        log_sa = load_run_log(hybrid_label)
        best_sa = build_logged_solution(log_sa, Tri)
        hist_sa = list(log_sa.get('fitness_history', []))

    hybrid_rmse = float(best_sa.fitness())
    print(f"  [worker] Hybrid seed {seed}: GA={ga_rmse:.4f} → SA={hybrid_rmse:.4f}", flush=True)
    return seed, ga_rmse, hybrid_rmse, hist_sa


def run_val_seed(seed, ga_cfg):
    """GA-only validation for one seed, offset +100 (Section 10/11).
    Returns: (seed, rmse, fitness_history)"""
    (Tri, load_or_run_ga, _, _,
     tournament_sel, uniform_xo, sp_xo, mut, _, partial) = _worker_imports()

    actual_seed = seed + 100
    random.seed(actual_seed)
    np.random.seed(actual_seed)

    xo, sel = _build_xo_sel(ga_cfg, uniform_xo, sp_xo, tournament_sel, partial)
    best, stats, _, _, _ = load_or_run_ga(
        f'final_validation_seed{seed}',
        solution_class=Tri, cfg=ga_cfg,
        selection_fn=sel, xo_fn=xo, mut_fn=mut, verbose=False,
    )
    rmse = best.fitness()
    print(f"  [worker] Val seed {seed}: RMSE={rmse:.4f}", flush=True)
    return seed, rmse, stats['fitness_history']


# ── new: generic single-seed worker (used by run_multi_seed) ─────────────────

def _multi_seed_worker(args):
    """Top-level picklable worker for run_multi_seed."""
    label, seed, cfg = args
    (Tri, load_or_run_ga, _, _,
     tournament_sel, uniform_xo, sp_xo, mut, _, partial) = _worker_imports()

    random.seed(seed)
    np.random.seed(seed)

    xo, sel = _build_xo_sel(cfg, uniform_xo, sp_xo, tournament_sel, partial)
    best, stats, _, _, _ = load_or_run_ga(
        f'{label}_seed{seed}',
        solution_class=Tri, cfg=cfg,
        selection_fn=sel, xo_fn=xo, mut_fn=mut, verbose=False,
    )
    rmse              = best.fitness()
    history           = stats['fitness_history']
    mean_dist_history = stats.get('mean_dist_history', [])
    print(f"  [worker] {label} seed {seed}: RMSE={rmse:.4f}", flush=True)
    return seed, rmse, history, mean_dist_history


def run_multi_seed(label, cfg, n_seeds=30, n_workers=None):
    """Run cfg across n_seeds in parallel, caching per (label, seed) via load_or_run_ga.

    Returns: {rmses, histories, mean_dist_histories, mean, sample_sd, config, seeds}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    rmses               = [None] * n_seeds
    histories           = [None] * n_seeds
    mean_dist_histories = [None] * n_seeds

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_multi_seed_worker, (label, s, cfg)): s
            for s in range(n_seeds)
        }
        for future in as_completed(futures):
            seed, rmse, hist, mdh = future.result()
            rmses[seed]               = float(rmse)
            histories[seed]           = hist
            mean_dist_histories[seed] = mdh

    arr = np.array(rmses, dtype=float)
    return {
        'rmses':               rmses,
        'histories':           histories,
        'mean_dist_histories': mean_dist_histories,
        'mean':                float(arr.mean()),
        'sample_sd':           float(arr.std(ddof=1)) if n_seeds > 1 else 0.0,
        'config':              cfg,
        'seeds':               list(range(n_seeds)),
    }


# ── new: paired-test worker (run every config for one seed) ──────────────────

def _all_configs_worker(args):
    """Top-level picklable worker for run_seed_all_configs.
    Runs every config in config_dict with the same seed so results are paired."""
    seed, config_dict = args
    (Tri, load_or_run_ga, _, _,
     tournament_sel, uniform_xo, sp_xo, mut, _, partial) = _worker_imports()

    results = {}
    for name, cfg in config_dict.items():
        random.seed(seed)
        np.random.seed(seed)
        xo, sel = _build_xo_sel(cfg, uniform_xo, sp_xo, tournament_sel, partial)
        best, stats, _, _, _ = load_or_run_ga(
            f'paired_{name}_seed{seed}',
            solution_class=Tri, cfg=cfg,
            selection_fn=sel, xo_fn=xo, mut_fn=mut, verbose=False,
        )
        results[name] = {
            'rmse':              float(best.fitness()),
            'history':           stats['fitness_history'],
            'mean_dist_history': stats.get('mean_dist_history', []),
        }
        print(f"  [worker] seed={seed} cfg={name}: RMSE={best.fitness():.4f}", flush=True)
    return seed, results


def run_seed_all_configs(config_dict, n_seeds=30, n_workers=None):
    """For each seed, run every config with the same RNG state.
    Enables paired Wilcoxon: each config sees identical random draws per seed.

    Returns: {cfg_name: {rmses: [...n_seeds], histories: [...], mean_dist_histories: [...]}}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    per_seed = [None] * n_seeds
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_all_configs_worker, (s, config_dict)): s
            for s in range(n_seeds)
        }
        for future in as_completed(futures):
            seed, seed_results = future.result()
            per_seed[seed] = seed_results

    # Transpose: per_seed[s][name] → by_config[name] lists of length n_seeds
    return {
        name: {
            'rmses':               [per_seed[s][name]['rmse']              for s in range(n_seeds)],
            'histories':           [per_seed[s][name]['history']           for s in range(n_seeds)],
            'mean_dist_histories': [per_seed[s][name]['mean_dist_history'] for s in range(n_seeds)],
        }
        for name in config_dict
    }


# ── statistical helpers ───────────────────────────────────────────────────────

def _bootstrap_mean_diff_ci(diffs, n_boot=10000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    boot  = np.array([
        np.mean(np.random.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def _bootstrap_indep_diff_ci(a, b, n_boot=10000, alpha=0.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    boot = np.array([
        np.mean(np.random.choice(a, size=len(a), replace=True)) -
        np.mean(np.random.choice(b, size=len(b), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def paired_wilcoxon(rmses_a, rmses_b, label_a='A', label_b='B'):
    """Paired Wilcoxon signed-rank test + 95% bootstrap CI on mean difference.

    Prints: '{label_a} vs {label_b}: ΔRMSE = X [95% CI L to U], paired Wilcoxon p = Y'
    Returns: {statistic, p_value, mean_diff, ci_lower, ci_upper, significant}
    """
    a, b  = np.asarray(rmses_a, dtype=float), np.asarray(rmses_b, dtype=float)
    diffs = a - b
    if np.all(diffs == 0):
        stat, pvalue = 0.0, 1.0
    else:
        stat, pvalue = scipy.stats.wilcoxon(a, b)
    mean_diff       = float(diffs.mean())
    ci_lo, ci_hi   = _bootstrap_mean_diff_ci(diffs)
    significant     = bool(pvalue < 0.05)
    print(f"{label_a} vs {label_b}: ΔRMSE = {mean_diff:+.4f} "
          f"[95% CI {ci_lo:+.4f} to {ci_hi:+.4f}], "
          f"paired Wilcoxon p = {pvalue:.4f}"
          + ("  ✓" if significant else ""))
    return {
        'statistic': float(stat), 'p_value': float(pvalue),
        'mean_diff': mean_diff, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
        'significant': significant,
    }


def unpaired_mannwhitney(rmses_a, rmses_b, label_a='A', label_b='B'):
    """Mann-Whitney U for independent samples. Same return shape as paired_wilcoxon."""
    a, b        = np.asarray(rmses_a, dtype=float), np.asarray(rmses_b, dtype=float)
    stat, pvalue = scipy.stats.mannwhitneyu(a, b, alternative='two-sided')
    mean_diff   = float(a.mean() - b.mean())
    ci_lo, ci_hi = _bootstrap_indep_diff_ci(a, b)
    significant  = bool(pvalue < 0.05)
    print(f"{label_a} vs {label_b}: ΔRMSE = {mean_diff:+.4f} "
          f"[95% CI {ci_lo:+.4f} to {ci_hi:+.4f}], "
          f"Mann-Whitney p = {pvalue:.4f}"
          + ("  ✓" if significant else ""))
    return {
        'statistic': float(stat), 'p_value': float(pvalue),
        'mean_diff': mean_diff, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
        'significant': significant,
    }


def screening_analysis(by_config, reference_name='optuna_500'):
    """Compute paired Wilcoxon + 95% bootstrap CI for each config vs a reference config.

    Args:
        by_config: output of run_seed_all_configs  — {name: {rmses, mean_dist_histories, ...}}
        reference_name: key in by_config used as the comparison baseline

    Returns: {cfg_name: {mean_diff, ci_95, p_value, mean_rmse, sample_sd,
                          significant, diversity_at_gen250}}
    """
    ref = np.asarray(by_config[reference_name]['rmses'], dtype=float)
    summary = {}
    for name, data in by_config.items():
        if name == reference_name:
            continue
        cfg_rmses = np.asarray(data['rmses'], dtype=float)
        diffs     = cfg_rmses - ref
        if np.all(diffs == 0):
            stat, pvalue = 0.0, 1.0
        else:
            stat, pvalue = scipy.stats.wilcoxon(cfg_rmses, ref)
        ci_lo, ci_hi = _bootstrap_mean_diff_ci(diffs)

        mdhs     = data['mean_dist_histories']
        div250   = float(np.mean([
            mdh[249] for mdh in mdhs if mdh and len(mdh) > 249
        ])) if any(mdh and len(mdh) > 249 for mdh in mdhs) else float('nan')

        summary[name] = {
            'mean_diff':          float(diffs.mean()),
            'ci_95':              (float(ci_lo), float(ci_hi)),
            'p_value':            float(pvalue),
            'mean_rmse':          float(cfg_rmses.mean()),
            'sample_sd':          float(cfg_rmses.std(ddof=1)),
            'significant':        bool(pvalue < 0.05),
            'diversity_at_gen250': div250,
        }
        sig = summary[name]
        print(f"  {name}: RMSE {sig['mean_rmse']:.4f}±{sig['sample_sd']:.4f}, "
              f"ΔRMSE={sig['mean_diff']:+.4f} [{ci_lo:+.4f} to {ci_hi:+.4f}], "
              f"p={pvalue:.4f}" + ("  ✓" if pvalue < 0.05 else ""))
    return summary


# ── OAT statistical analysis ──────────────────────────────────────────────────

def _cliffs_delta(a, b):
    """Cliff's delta effect size in [-1, 1]. |δ| < 0.147 negligible, < 0.33 small."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def _holm_correct(pvalues):
    """Holm step-down correction. Returns adjusted p-values in original order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj[idx] = min(max(running_max, p[idx] * (n - rank)), 1.0)
        running_max = adj[idx]
    return adj.tolist()


def analyze_oat(results, param_name, param_type='numeric', alpha=0.05,
                baseline_rmses=None, correct='holm'):
    """Statistical analysis of one OAT experiment.

    Parameters
    ----------
    results : list of (value, mean_rmse, sd_rmse, all_rmses)
        Exact output of run_oat_experiment.
    param_type : 'numeric' or 'categorical'
    baseline_rmses : optional list of RMSEs from an independent baseline run.
        Compared against the best value using unpaired Mann-Whitney U.

    Returns dict with equivalence groups, adjusted p-values, Cliff's delta,
    and a vs_baseline entry.
    """
    k = len(results)
    values     = [r[0] for r in results]
    means      = [r[1] for r in results]
    rmse_lists = [list(r[3]) for r in results]
    best_idx   = int(np.argmin(means))
    best_value = values[best_idx]
    best_rmses = np.asarray(rmse_lists[best_idx], dtype=float)

    # Sentinels for best index slot (no self-comparison)
    raw_pvals_vs_best   = [float('nan')] * k
    adj_pvals_vs_best   = [float('nan')] * k
    cliffs_delta_vs_best = [0.0] * k
    mean_diff_vs_best   = [0.0] * k

    non_best = [i for i in range(k) if i != best_idx]
    raw_p_list = []
    for i in non_best:
        a = np.asarray(rmse_lists[i], dtype=float)
        diffs = a - best_rmses
        if np.all(diffs == 0):
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = scipy.stats.wilcoxon(a, best_rmses)
            except ValueError:
                stat, pval = 0.0, 1.0
        raw_pvals_vs_best[i]    = float(pval)
        cliffs_delta_vs_best[i] = _cliffs_delta(a, best_rmses)
        mean_diff_vs_best[i]    = float(a.mean() - best_rmses.mean())
        raw_p_list.append(float(pval))

    adj_list = _holm_correct(raw_p_list) if correct == 'holm' else (
        [min(p * len(raw_p_list), 1.0) for p in raw_p_list] if correct == 'bonferroni'
        else raw_p_list
    )
    for j, i in enumerate(non_best):
        adj_pvals_vs_best[i] = adj_list[j]

    equiv_idx = [i for i in range(k)
                 if i == best_idx or (not np.isnan(adj_pvals_vs_best[i])
                                      and adj_pvals_vs_best[i] > alpha)]
    equiv_vals = [values[i] for i in equiv_idx]

    vs_baseline = None
    if baseline_rmses is not None:
        vs_baseline = unpaired_mannwhitney(
            best_rmses, np.asarray(baseline_rmses, dtype=float),
            str(best_value), 'baseline',
        )

    return {
        'param_name': param_name,
        'param_type': param_type,
        'values': values,
        'means': means,
        'rmse_lists': rmse_lists,
        'best_idx': best_idx,
        'best_value': best_value,
        'raw_pvals_vs_best': raw_pvals_vs_best,
        'adj_pvals_vs_best': adj_pvals_vs_best,
        'cliffs_delta_vs_best': cliffs_delta_vs_best,
        'mean_diff_vs_best': mean_diff_vs_best,
        'equivalent_to_best': equiv_vals,
        'equivalent_to_best_idx': equiv_idx,
        'vs_baseline': vs_baseline,
        'alpha': alpha,
        'correction': correct,
    }


def recommend_range(analysis, extend_pct=0.3, lower_floor=1):
    """Derive an Optuna search range from the equivalence group."""
    eq_idx  = analysis['equivalent_to_best_idx']
    values  = analysis['values']
    best_idx = analysis['best_idx']
    k = len(values)

    if analysis['param_type'] == 'categorical':
        return {
            'type': 'categorical',
            'choices': [values[i] for i in eq_idx],
            'note': f"{len(eq_idx)}/{k} choices equiv to best ({analysis['best_value']})",
        }

    eq_vals = [values[i] for i in eq_idx]
    low, high = min(eq_vals), max(eq_vals)
    notes = [f"{len(eq_vals)}/{k} values equiv to best ({analysis['best_value']})"]

    if best_idx == 0:
        new_low = max(lower_floor, low * (1 - extend_pct))
        notes.append(f"best at lower edge — extended low {low} → {new_low:.4g}")
        low = new_low
    if best_idx == k - 1:
        new_high = high * (1 + extend_pct)
        notes.append(f"best at upper edge — extended high {high} → {new_high:.4g}")
        high = new_high

    low = min(low, high)
    if all(isinstance(v, int) for v in values):
        return {'type': 'int', 'low': int(round(low)), 'high': int(round(high)), 'notes': notes}
    return {'type': 'float', 'low': float(low), 'high': float(high), 'notes': notes}


def print_oat_report(analysis):
    """Print per-value statistical table and recommendation. Returns recommendation dict."""
    a = analysis
    print(f"\n── OAT: {a['param_name']}  (best={a['best_value']},"
          f" correction={a['correction']}, α={a['alpha']}) ──")
    print(f"{'Value':<16} {'Mean':>8} {'Δ vs best':>10} {'Adj. p':>9} {'Cliff δ':>9}  Verdict")
    print("─" * 68)
    for i, v in enumerate(a['values']):
        mean_s  = f"{a['means'][i]:.4f}"
        diff_s  = f"{a['mean_diff_vs_best'][i]:+.4f}"
        adjp    = a['adj_pvals_vs_best'][i]
        adjp_s  = "N/A    " if np.isnan(adjp) else f"{adjp:.4f}"
        delta_s = f"{a['cliffs_delta_vs_best'][i]:+.3f}"
        if i == a['best_idx']:
            verdict = "BEST"
        elif i in a['equivalent_to_best_idx']:
            verdict = "equiv"
        else:
            verdict = "worse"
        print(f"{str(v):<16} {mean_s:>8} {diff_s:>10} {adjp_s:>9} {delta_s:>9}  {verdict}")

    rec = recommend_range(analysis)
    print(f"\nRecommendation → {rec}")
    if a['vs_baseline'] is not None:
        vb = a['vs_baseline']
        print(f"vs baseline: ΔRMSE={vb['mean_diff']:+.4f}, Mann-Whitney p={vb['p_value']:.4f}"
              + ("  ✓" if vb['significant'] else ""))
    return rec


def plot_oat_with_groups(analysis, baseline_rmse=None, ax=None):
    """Box plot colored by equivalence-to-best: dark green=best, light green=equiv, red=worse.

    Does not call plt.show() or plt.close() — caller controls display.
    Returns the Axes object.
    """
    a = analysis
    k = len(a['values'])
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, k * 1.3), 4.5))

    bp = ax.boxplot(a['rmse_lists'], patch_artist=True, widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='k',
                                   markeredgecolor='k', markersize=4))
    colors = []
    for i in range(k):
        if i == a['best_idx']:
            colors.append('#2ca02c')
        elif i in a['equivalent_to_best_idx']:
            colors.append('#98df8a')
        else:
            colors.append('#d62728')
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels([str(v) for v in a['values']], rotation=20 if k > 4 else 0)
    if baseline_rmse is not None:
        ax.axhline(baseline_rmse, color='tomato', ls='--', lw=1.2,
                   label=f'Baseline@300 ({baseline_rmse:.2f})')
        ax.legend(fontsize=9)
    ax.set_xlabel(a['param_name'])
    ax.set_ylabel('Final RMSE (30 seeds)')
    ax.set_title(f"{a['param_name']} — dark green=best, light green=equiv (Holm α=0.05), red=worse")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return ax
