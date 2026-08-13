"""
Central configuration — single source of truth for all hyperparameters.

The pipeline flows left to right through this file:
  GA_CONFIG  →  OAT grids  →  OPTUNA_SEARCH_SPACE  →  OPTUNA_BEST_CONFIG
             →  LAYER2/3 overrides  →  FINAL_BEST_CONFIG

Nothing is computed here; only values are declared.
"""

from datetime import datetime


# ── Baseline (un-tuned starting point, used in Sections 5–6) ──────────────────
GA_CONFIG = {
    'population_size': 30,
    'max_generations': 1000,
    'tournament_size': 3,
    'elitesize': 1,
    'xo_prob': 0.8,
    'mut_prob': 0.05,
    'crossover_type': 'uniform',
    'mutation_type': 'random',
    'maximization': False,
    'track_diversity': True,         # ALWAYS on — diagnostic is free
    'save_image_every_n_gens': 100,
}

# ── SA config for Challenge 2 (Section 9) ────────────────────────────────────
SA_CONFIG = {
    'C': 20.0, 'L': 150, 'H': 1.05,
    'max_iter': 200,                 # 150 × 200 = 30,000 = GA budget
    'save_every_n_steps': 1000,
}

# ── SA config for Section 10 hybrid GA → SA refinement ───────────────────────
# Lower C (start closer to GA's optimum) and faster cooling (H higher).
# Budget: L × max_iter = 100 × 60 = 6,000 evals ≈ 20% of 30,000 total.
HYBRID_SA_CONFIG = {
    'C': 10.0, 'L': 100, 'H': 1.05,
    'max_iter': 60,                  # 100 × 60 = 6,000 evals — higher C lets SA escape GA's local minimum
    'save_every_n_steps': 500,
}

# ── Section 6 OAT experiment grids ───────────────────────────────────────────
EXPERIMENT_A_POPULATION = [10, 20, 50]
EXPERIMENT_B_ELITE      = [0, 1, 5]
EXPERIMENT_C_MUTRATE    = [0.01, 0.05, 0.1, 0.2]
EXPERIMENT_D_CROSSOVER  = ['uniform', 'single_point']
EXPERIMENT_E_MUTTYPE    = ['vertex_shift', 'triangle_replace', 'random']
EXPERIMENT_F_TOURNAMENT = [2, 3, 4, 5, 6]
EXPERIMENT_G_XOPROB     = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXPERIMENT_H_INVERTED   = [
    {'mut_prob': 0.10, 'xo_prob': 0.6},
    {'mut_prob': 0.15, 'xo_prob': 0.6},
    {'mut_prob': 0.20, 'xo_prob': 0.5},
]
EXPERIMENT_I_STEPSIZE   = [
    {'vertex_step': 10, 'color_step': 15, 'alpha_step': 15},
    {'vertex_step': 30, 'color_step': 30, 'alpha_step': 30},   # default
    {'vertex_step': 50, 'color_step': 80, 'alpha_step': 50},
]

# ── Optuna search space (filled by Section 6 derivation cell) ─────────────────
OPTUNA_SEARCH_SPACE = {'population_size': {'type': 'int', 'low': 50, 'high': 65, 'notes': ['1/3 values equiv to best (50)', 'best at upper edge — extended high 50 → 65']}, 'elitesize': {'type': 'int', 'low': 1, 'high': 5, 'notes': ['3/3 values equiv to best (0)', 'best at lower edge — extended low 0 → 1']}, 'mut_prob': {'type': 'float', 'low': 0.1, 'high': 0.2, 'notes': ['2/4 values equiv to best (0.1)']}, 'crossover_type': {'type': 'categorical', 'choices': ['uniform'], 'note': '1/2 choices equiv to best (uniform)'}, 'mutation_type': {'type': 'categorical', 'choices': ['vertex_shift', 'triangle_replace', 'random'], 'note': '3/3 choices equiv to best (vertex_shift)'}, 'tournament_size': {'type': 'int', 'low': 4, 'high': 8, 'notes': ['2/5 values equiv to best (6)', 'best at upper edge — extended high 6 → 7.8']}, 'xo_prob': {'type': 'float', 'low': 0.9, 'high': 1.3, 'notes': ['2/6 values equiv to best (1.0)', 'best at upper edge — extended high 1.0 → 1.3']}, 'vertex_step': {'type': 'int', 'low': 50, 'high': 65}, 'color_step': {'type': 'int', 'low': 80, 'high': 104}, 'alpha_step': {'type': 'int', 'low': 50, 'high': 65}}
OPTUNA_BEST_CONFIG = {'population_size': 63, 'max_generations': 1000, 'tournament_size': 7, 'elitesize': 4, 'xo_prob': 1.0553931118820694, 'mut_prob': 0.1016390739500758, 'crossover_type': 'uniform', 'mutation_type': 'triangle_replace', 'maximization': False, 'track_diversity': True, 'save_image_every_n_gens': 100, 'vertex_step': 58, 'color_step': 91, 'alpha_step': 50}
LAYER2_GEOMETRIC       = {'mutation_type': 'geometric'}
LAYER2_REPLACE         = {'innovation_replace_prob': 0.02}
LAYER2_SWAP            = {'zorder_swap_prob': 0.01}
LAYER3_FITNESS_SHARING = {'fitness_sharing': True, 'sigma_share': 0.1}
LAYER3_DYNAMIC         = {
    'dynamic_schedule': True,
    'mut_prob_start': 0.15, 'mut_prob_end': 0.03,
    'tournament_start': 2,  'tournament_end': 5,
}

# ── Final combined config (filled after Section 8.6 picks winner) ─────────────
FINAL_BEST_CONFIG = {'population_size': 63, 'max_generations': 1000, 'tournament_size': 7, 'elitesize': 4, 'xo_prob': 1.0553931118820694, 'mut_prob': 0.1016390739500758, 'crossover_type': 'uniform', 'mutation_type': 'triangle_replace', 'maximization': False, 'track_diversity': True, 'save_image_every_n_gens': 100, 'vertex_step': 58, 'color_step': 91, 'alpha_step': 50, 'dynamic_schedule': True, 'mut_prob_start': 0.15, 'mut_prob_end': 0.03, 'tournament_start': 2, 'tournament_end': 5}




















def make_run_id(label):
    return f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
