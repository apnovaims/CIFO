"""
Mutation operators for the Triangle Image GA.

Seven operator modes are supported.  The original five (vertex_shift,
color_shift, alpha_shift, triangle_replace, triangle_swap) are unchanged in
behaviour but now accept configurable step sizes.  Two new modes are added:

    "geometric"  — box mutation: perturbs ALL 10 genes of a triangle
                   atomically using type-specific step sizes.
    "mixed"      — same as geometric but applies each gene type's own shift
                   logic independently (vertex_shift per vertex gene,
                   color_shift per RGB gene, alpha_shift per alpha gene).

Two background innovation operators fire independently AFTER the main loop:

    innovation_replace_prob  — per-triangle probability of full randomisation
    zorder_swap_prob         — per-individual probability of one z-order swap

Step sizes (vertex_step, color_step, alpha_step) default to 30, matching
the original hardcoded ±30 behaviour.
"""

import random


_GENES_PER_TRIANGLE = 10


def triangle_mutation(
    individual,
    mut_prob: float,
    mutation_type: str = "random",
    verbose: bool = False,
    vertex_step: int = 30,
    color_step: int = 30,
    alpha_step: int = 30,
    innovation_replace_prob: float = 0.0,
    zorder_swap_prob: float = 0.0,
):
    """
    Apply per-triangle mutation to an individual.

    For each of the 100 triangles, with probability mut_prob, one mutation
    operator fires.  The operator is selected by mutation_type.

    Parameters
    ----------
    individual : TriangleImageSolution
    mut_prob : float — probability that any given triangle is mutated.
    mutation_type : str — one of:
        "vertex_shift"     small displacement of one vertex (±vertex_step px)
        "color_shift"      small change to one RGB channel (±color_step)
        "alpha_shift"      small change to alpha / transparency (±alpha_step)
        "triangle_replace" replace the entire triangle with a random one
        "triangle_swap"    swap the z-order of two random triangles
        "random"           choose uniformly from vertex_shift, color_shift,
                           alpha_shift (exploitation-only)
        "geometric"        box mutation — perturb ALL 10 genes atomically
        "mixed"            geometric-style but applies each gene's own shift
                           operator independently
    vertex_step : int  — step size for vertex coordinate perturbations (default 30)
    color_step  : int  — step size for RGB channel perturbations (default 30)
    alpha_step  : int  — step size for alpha perturbations (default 30)
    innovation_replace_prob : float — per-triangle probability of full randomisation
                                      applied AFTER the main mutation loop
    zorder_swap_prob        : float — per-individual probability of one z-order swap
                                      applied AFTER the main mutation loop
    verbose : bool

    Returns
    -------
    TriangleImageSolution — a new mutated individual.
    """
    img_w = individual.__class__.IMG_W
    img_h = individual.__class__.IMG_H
    n_triangles = len(individual.repr) // _GENES_PER_TRIANGLE

    new_repr = individual.repr.copy()

    _EXPLOITATION_OPS = ["vertex_shift", "color_shift", "alpha_shift"]

    i = 0
    while i < n_triangles:
        if random.random() > mut_prob:
            i += 1
            continue

        op = random.choice(_EXPLOITATION_OPS) if mutation_type == "random" else mutation_type

        # ── z-order swap (acts on two triangles, skip both) ──────────────────
        if op == "triangle_swap":
            j = random.randint(0, n_triangles - 1)
            if j == i:
                j = (i + 1) % n_triangles
            base_i = i * _GENES_PER_TRIANGLE
            base_j = j * _GENES_PER_TRIANGLE
            (new_repr[base_i:base_i + _GENES_PER_TRIANGLE],
             new_repr[base_j:base_j + _GENES_PER_TRIANGLE]) = (
                new_repr[base_j:base_j + _GENES_PER_TRIANGLE],
                new_repr[base_i:base_i + _GENES_PER_TRIANGLE],
            )
            if verbose:
                print(f"triangle_swap: triangles {i} and {j} swapped z-order.")
            i += 1
            continue

        base = i * _GENES_PER_TRIANGLE

        # ── single-gene exploitation operators ───────────────────────────────
        if op == "vertex_shift":
            vertex = random.randint(0, 2)
            axis   = random.randint(0, 1)
            gene   = base + vertex * 2 + axis
            bound  = img_w if axis == 0 else img_h
            delta  = random.randint(-vertex_step, vertex_step)
            new_repr[gene] = max(0, min(bound, new_repr[gene] + delta))
            if verbose:
                print(f"vertex_shift: triangle {i}, vertex {vertex}, "
                      f"axis {'x' if axis==0 else 'y'}, delta={delta}")

        elif op == "color_shift":
            channel = random.randint(0, 2)
            gene    = base + 6 + channel
            delta   = random.randint(-color_step, color_step)
            new_repr[gene] = max(0, min(255, new_repr[gene] + delta))
            if verbose:
                print(f"color_shift: triangle {i}, channel {['R','G','B'][channel]}, delta={delta}")

        elif op == "alpha_shift":
            gene  = base + 9
            delta = random.randint(-alpha_step, alpha_step)
            new_repr[gene] = max(0, min(255, new_repr[gene] + delta))
            if verbose:
                print(f"alpha_shift: triangle {i}, delta={delta}")

        # ── full replacement ─────────────────────────────────────────────────
        elif op == "triangle_replace":
            new_repr[base]     = random.randint(0, img_w)
            new_repr[base + 1] = random.randint(0, img_h)
            new_repr[base + 2] = random.randint(0, img_w)
            new_repr[base + 3] = random.randint(0, img_h)
            new_repr[base + 4] = random.randint(0, img_w)
            new_repr[base + 5] = random.randint(0, img_h)
            new_repr[base + 6] = random.randint(0, 255)
            new_repr[base + 7] = random.randint(0, 255)
            new_repr[base + 8] = random.randint(0, 255)
            new_repr[base + 9] = random.randint(0, 255)
            if verbose:
                print(f"triangle_replace: triangle {i} fully randomised.")

        # ── geometric (box) mutation — all 10 genes atomically ──────────────
        elif op == "geometric":
            for j in range(6):   # vertex coordinates
                bound = img_w if j % 2 == 0 else img_h
                delta = random.randint(-vertex_step, vertex_step)
                new_repr[base + j] = max(0, min(bound, new_repr[base + j] + delta))
            for j in range(6, 9):   # RGB
                delta = random.randint(-color_step, color_step)
                new_repr[base + j] = max(0, min(255, new_repr[base + j] + delta))
            delta = random.randint(-alpha_step, alpha_step)
            new_repr[base + 9] = max(0, min(255, new_repr[base + 9] + delta))
            if verbose:
                print(f"geometric: triangle {i} all 10 genes perturbed.")

        # ── mixed: per-gene independent operator logic on all 10 genes ───────
        elif op == "mixed":
            for j in range(6):
                bound = img_w if j % 2 == 0 else img_h
                delta = random.randint(-vertex_step, vertex_step)
                new_repr[base + j] = max(0, min(bound, new_repr[base + j] + delta))
            for j in range(6, 9):
                delta = random.randint(-color_step, color_step)
                new_repr[base + j] = max(0, min(255, new_repr[base + j] + delta))
            delta = random.randint(-alpha_step, alpha_step)
            new_repr[base + 9] = max(0, min(255, new_repr[base + 9] + delta))
            if verbose:
                print(f"mixed: triangle {i} all genes perturbed independently.")

        i += 1

    # ── background innovation operators ──────────────────────────────────────
    if innovation_replace_prob > 0.0:
        for i in range(n_triangles):
            if random.random() < innovation_replace_prob:
                base = i * _GENES_PER_TRIANGLE
                new_repr[base]     = random.randint(0, img_w)
                new_repr[base + 1] = random.randint(0, img_h)
                new_repr[base + 2] = random.randint(0, img_w)
                new_repr[base + 3] = random.randint(0, img_h)
                new_repr[base + 4] = random.randint(0, img_w)
                new_repr[base + 5] = random.randint(0, img_h)
                new_repr[base + 6] = random.randint(0, 255)
                new_repr[base + 7] = random.randint(0, 255)
                new_repr[base + 8] = random.randint(0, 255)
                new_repr[base + 9] = random.randint(0, 255)
                if verbose:
                    print(f"background innovation_replace_prob: triangle {i} fully randomised.")

    if zorder_swap_prob > 0.0 and random.random() < zorder_swap_prob:
        i, j = random.sample(range(n_triangles), 2)
        base_i, base_j = i * _GENES_PER_TRIANGLE, j * _GENES_PER_TRIANGLE
        (new_repr[base_i:base_i + _GENES_PER_TRIANGLE],
         new_repr[base_j:base_j + _GENES_PER_TRIANGLE]) = (
            new_repr[base_j:base_j + _GENES_PER_TRIANGLE],
            new_repr[base_i:base_i + _GENES_PER_TRIANGLE],
        )
        if verbose:
            print(f"background zorder_swap_prob: triangles {i} and {j} z-order swapped.")

    return individual.with_repr(new_repr)
