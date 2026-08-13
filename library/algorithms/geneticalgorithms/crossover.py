"""
Crossover operators for the Triangle Image GA.

Why not binary_standard_crossover from P08?
-------------------------------------------
binary_standard_crossover operates on lists of bits and calls binaryrepr().
Our representation is a flat list of integers at the *triangle* level (100
triangles × 10 genes = 1000 integers).  Crossing over at an arbitrary bit
boundary would split a triangle's genes in the middle, producing malformed
triangles.  We therefore define crossover at the triangle boundary.

Two operators are provided so their effects can be compared experimentally
(see notebook Section 6 — hyperparameter experiments).

Both operators respect the with_repr() interface established in P09 so they
work with any Solution subclass that implements with_repr().
"""

import random
from copy import deepcopy


# Number of integers per triangle (constant — do not change unless repr changes)
_GENES_PER_TRIANGLE = 10


def block_triangle_crossover(parent1, parent2, crossover_prob: float, verbose: bool = False):
    """
    Block triangle crossover — z-order-aware.

    Inherits a contiguous block of triangle z-positions [a, b) from one parent
    and the rest from the other.  Unlike uniform crossover (which independently
    assigns each triangle slot), this preserves z-order coherence within each
    inherited block — relevant for alpha-blending epistasis where the rendering
    order of a group of semi-transparent triangles matters.

    offspring1 = parent2's triangles with block [a,b) replaced by parent1's
    offspring2 = parent1's triangles with block [a,b) replaced by parent2's

    Parameters
    ----------
    parent1, parent2 : TriangleImageSolution
    crossover_prob : float
    verbose : bool

    Returns
    -------
    (offspring1, offspring2) : tuple[TriangleImageSolution, TriangleImageSolution]
    """
    n_triangles = len(parent1.repr) // _GENES_PER_TRIANGLE

    if random.random() > crossover_prob:
        if verbose:
            print("No crossover — returning copies of parents.")
        return deepcopy(parent1), deepcopy(parent2)

    a = random.randint(0, n_triangles - 2)
    b = random.randint(a + 1, n_triangles)

    repr1 = parent2.repr.copy()
    repr1[a * _GENES_PER_TRIANGLE: b * _GENES_PER_TRIANGLE] = \
        parent1.repr[a * _GENES_PER_TRIANGLE: b * _GENES_PER_TRIANGLE]

    repr2 = parent1.repr.copy()
    repr2[a * _GENES_PER_TRIANGLE: b * _GENES_PER_TRIANGLE] = \
        parent2.repr[a * _GENES_PER_TRIANGLE: b * _GENES_PER_TRIANGLE]

    if verbose:
        print(f"Block crossover: block [{a}, {b}) "
              f"(genes {a*_GENES_PER_TRIANGLE}–{b*_GENES_PER_TRIANGLE-1}).")

    return parent1.with_repr(repr1), parent2.with_repr(repr2)


def uniform_triangle_crossover(parent1, parent2, crossover_prob: float, verbose: bool = False):
    """
    Uniform triangle crossover.

    For each of the N triangles, independently draw from parent1 or parent2
    with equal probability (50/50).  The resulting offspring are a random
    mosaic of triangles from both parents.

    This operator maximally mixes genetic material — useful when parents
    represent very different areas of the image and we want to explore the
    space between them.

    Parameters
    ----------
    parent1, parent2 : TriangleImageSolution
    crossover_prob : float  — probability that crossover actually occurs.
                             If it does not occur, offspring are copies of parents.
    verbose : bool

    Returns
    -------
    (offspring1, offspring2) : tuple[TriangleImageSolution, TriangleImageSolution]
    """
    n_triangles = len(parent1.repr) // _GENES_PER_TRIANGLE

    if random.random() <= crossover_prob:
        repr1 = parent1.repr.copy()
        repr2 = parent2.repr.copy()
        new_repr1 = []
        new_repr2 = []

        for i in range(n_triangles):
            start = i * _GENES_PER_TRIANGLE
            end   = start + _GENES_PER_TRIANGLE
            if random.random() < 0.5:
                new_repr1.extend(repr1[start:end])
                new_repr2.extend(repr2[start:end])
            else:
                new_repr1.extend(repr2[start:end])
                new_repr2.extend(repr1[start:end])

        if verbose:
            print(f"Uniform crossover performed across {n_triangles} triangles.")

        return parent1.with_repr(new_repr1), parent2.with_repr(new_repr2)

    else:
        if verbose:
            print("No crossover — returning copies of parents.")
        return deepcopy(parent1), deepcopy(parent2)


def single_point_triangle_crossover(parent1, parent2, crossover_prob: float, verbose: bool = False):
    """
    Single-point triangle crossover.

    Chooses a random cut point K in [1, N-1] (triangle index).
    offspring1 = parent1's triangles 0…K-1  +  parent2's triangles K…N-1
    offspring2 = parent2's triangles 0…K-1  +  parent1's triangles K…N-1

    This is the triangle-level analogue of the bit-level single-point
    crossover from P08.  It preserves spatial coherence within each half —
    useful if one parent has a good top half and another has a good bottom half.

    Parameters
    ----------
    parent1, parent2 : TriangleImageSolution
    crossover_prob : float
    verbose : bool

    Returns
    -------
    (offspring1, offspring2) : tuple[TriangleImageSolution, TriangleImageSolution]
    """
    n_triangles = len(parent1.repr) // _GENES_PER_TRIANGLE

    if random.random() <= crossover_prob:
        cut = random.randint(1, n_triangles - 1)  # triangle index of the cut
        cut_gene = cut * _GENES_PER_TRIANGLE       # corresponding flat-list index

        repr1 = parent1.repr
        repr2 = parent2.repr

        new_repr1 = repr1[:cut_gene] + repr2[cut_gene:]
        new_repr2 = repr2[:cut_gene] + repr1[cut_gene:]

        if verbose:
            print(f"Single-point crossover at triangle {cut} (gene index {cut_gene}).")

        return parent1.with_repr(new_repr1), parent2.with_repr(new_repr2)

    else:
        if verbose:
            print("No crossover — returning copies of parents.")
        return deepcopy(parent1), deepcopy(parent2)
