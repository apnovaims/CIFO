"""
TriangleImageSolution
=====================
Represents a candidate image composed of N_TRIANGLES semi-transparent colored
triangles drawn over a black canvas.  The goal is to minimize the pixel-wise
Root Mean Square Error (RMSE) between the rendered image and a target photograph.

Representation
--------------
A flat Python list of length N_TRIANGLES * 10.
Each consecutive block of 10 integers encodes one triangle:

    index  | field  | range
    -------|--------|-------
    0      | x1     | [0, IMG_W]
    1      | y1     | [0, IMG_H]
    2      | x2     | [0, IMG_W]
    3      | y2     | [0, IMG_H]
    4      | x3     | [0, IMG_W]
    5      | y3     | [0, IMG_H]
    6      | R      | [0, 255]
    7      | G      | [0, 255]
    8      | B      | [0, 255]
    9      | A      | [20, 180] (initialised; avoids fully transparent or fully opaque)

Triangle 0 is painted first (bottom layer); triangle 99 is on top.

Design decisions
----------------
- Alpha channel allows triangles to blend softly — important for smooth tonal
  gradations in portrait painting.
- RGBA compositing is done with PIL's Image.alpha_composite so overlapping
  triangles interact naturally.
- Fitness is cached in self._fitness after first evaluation; with_repr() always
  creates a fresh instance with _fitness=None so stale values are never reused.
"""

import random
import numpy as np
from PIL import Image, ImageDraw
from .solution import Solution


class TriangleImageSolution(Solution):

    N_TRIANGLES = 100
    IMG_W = 300
    IMG_H = 400

    # Class-level reference to the target image (numpy array, shape H×W×3, uint8).
    # Set once via TriangleImageSolution.load_target(path) before creating any instance.
    target_array = None

    # Pre-computed bounds list reused in random_initial_representation and clamping.
    # Pattern: [x_bound, y_bound, x_bound, y_bound, x_bound, y_bound, 255, 255, 255, 255]
    _BOUNDS = None

    @classmethod
    def load_target(cls, image_path: str):
        """Load the target image from disk and store it as a class attribute."""
        img = Image.open(image_path).convert("RGB").resize((cls.IMG_W, cls.IMG_H))
        cls.target_array = np.array(img, dtype=np.float32)
        cls._build_bounds()

    @classmethod
    def _build_bounds(cls):
        """Pre-compute the per-gene upper bounds list (used in random init and clamping)."""
        triangle_bounds = [cls.IMG_W, cls.IMG_H,  # x1, y1
                           cls.IMG_W, cls.IMG_H,  # x2, y2
                           cls.IMG_W, cls.IMG_H,  # x3, y3
                           255, 255, 255, 255]     # R, G, B, A
        cls._BOUNDS = triangle_bounds * cls.N_TRIANGLES

    # ── Construction ────────────────────────────────────────────────────────

    def __init__(self, repr=None, target_array=None):
        """
        Parameters
        ----------
        repr : list[int] | None
            Flat list of 1000 integers.  If None a random one is generated.
        target_array : ignored
            Kept for API symmetry; the class attribute is used instead.
        """
        self._fitness = None
        super().__init__(repr)

    def random_initial_representation(self):
        """Return a flat list of 1000 random integers within the gene bounds."""
        if self._BOUNDS is None:
            self._build_bounds()
        repr_list = [random.randint(0, b) for b in self._BOUNDS]
        # Alpha initialised to [20, 180] so every triangle contributes visibly
        # from generation 1 without being fully opaque and blocking layers below.
        for i in range(self.N_TRIANGLES):
            repr_list[i * 10 + 9] = random.randint(20, 180)
        return repr_list

    # ── GA interface ─────────────────────────────────────────────────────────

    def with_repr(self, new_repr):
        """
        Factory method required by crossover and mutation.

        Crossover and mutation produce a new flat list of 1000 integers.
        They call with_repr() to wrap that list into a proper Solution object,
        inheriting the class-level target image without re-loading it.
        The new instance starts with _fitness=None so fitness is re-computed
        on demand rather than carrying over a stale cached value.
        """
        return TriangleImageSolution(repr=list(new_repr))

    # ── Image rendering ───────────────────────────────────────────────────────

    def draw_image(self) -> np.ndarray:
        """
        Paint all 100 triangles onto a black RGBA canvas and return an RGB
        numpy array of shape (IMG_H, IMG_W, 3).

        Each triangle is composited individually so that semi-transparent
        triangles blend correctly with all layers beneath them.  Drawing all
        triangles on one shared layer before compositing would cause them to
        overwrite each other instead of blending (Porter-Duff Over operator
        requires per-triangle compositing).
        """
        canvas = Image.new("RGBA", (self.IMG_W, self.IMG_H), (0, 0, 0, 255))
        genes = self.repr
        for i in range(self.N_TRIANGLES):
            base = i * 10
            x1, y1 = genes[base],     genes[base + 1]
            x2, y2 = genes[base + 2], genes[base + 3]
            x3, y3 = genes[base + 4], genes[base + 5]
            r, g, b, a = genes[base + 6], genes[base + 7], genes[base + 8], genes[base + 9]
            layer = Image.new("RGBA", (self.IMG_W, self.IMG_H), (0, 0, 0, 0))
            ImageDraw.Draw(layer).polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(r, g, b, a))
            canvas = Image.alpha_composite(canvas, layer)
        return np.array(canvas.convert("RGB"), dtype=np.float32)

    # ── Fitness ───────────────────────────────────────────────────────────────

    def fitness(self) -> float:
        """
        Pixel-wise RMSE between the rendered image and the target.

        Lower is better (minimization problem).
        Result is cached so repeated calls within the same algorithm step
        are free.  with_repr() always creates a new instance with _fitness=None.
        """
        if self._fitness is not None:
            return self._fitness

        if self.target_array is None:
            raise RuntimeError(
                "Target image not loaded. Call TriangleImageSolution.load_target(path) first."
            )

        rendered = self.draw_image()
        diff = rendered - self.target_array
        self._fitness = float(np.sqrt(np.mean(diff ** 2)))
        return self._fitness

    # ── SA interface ──────────────────────────────────────────────────────────

    def get_random_neighbor(self):
        """
        Return a new solution that differs from self by one small mutation
        applied to one random triangle.

        Required by simulated_annealing() from the P07 library.
        The mutation type is chosen randomly from a subset of lightweight
        operators (vertex_shift, color_shift, alpha_shift) — we avoid
        triangle_replace here because SA already explores aggressively via
        its temperature schedule.
        """
        new_repr = self.repr.copy()
        tri_idx = random.randint(0, self.N_TRIANGLES - 1)
        base = tri_idx * 10
        mutation = random.choice(["vertex_shift", "color_shift", "alpha_shift"])

        if mutation == "vertex_shift":
            vertex = random.randint(0, 2)          # which of the 3 vertices
            axis   = random.randint(0, 1)           # x or y
            gene   = base + vertex * 2 + axis
            bound  = self.IMG_W if axis == 0 else self.IMG_H
            delta  = random.randint(-30, 30)
            new_repr[gene] = max(0, min(bound, new_repr[gene] + delta))

        elif mutation == "color_shift":
            channel = random.randint(0, 2)          # R, G, or B
            gene = base + 6 + channel
            delta = random.randint(-30, 30)
            new_repr[gene] = max(0, min(255, new_repr[gene] + delta))

        else:  # alpha_shift
            gene = base + 9
            delta = random.randint(-30, 30)
            new_repr[gene] = max(0, min(255, new_repr[gene] + delta))

        return TriangleImageSolution(repr=new_repr)

    # ── Display ───────────────────────────────────────────────────────────────

    def show(self, title: str = None, ax=None):
        """
        Display the rendered image using matplotlib.

        Parameters
        ----------
        title : str | None
            Optional title override.  If None, shows the fitness value.
        ax : matplotlib.axes.Axes | None
            If provided, draws into this axes object; otherwise creates a new figure.
        """
        import matplotlib.pyplot as plt

        img = self.draw_image().astype(np.uint8)
        fitness_str = f"RMSE = {self.fitness():.4f}"

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 5))
            standalone = True
        else:
            standalone = False

        ax.imshow(img)
        ax.set_title(title if title else fitness_str)
        ax.axis("off")

        if standalone:
            plt.tight_layout()
            plt.show()

    def __repr__(self):
        fitness_str = f"{self._fitness:.4f}" if self._fitness is not None else "not computed"
        return f"TriangleImageSolution(fitness={fitness_str})"
