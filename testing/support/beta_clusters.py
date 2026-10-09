"""A bare-bones beta-criterion clustering of a range image, only for the tests of the ground removal.

It exists so test_ground.py can show what happens to ground rows that are not removed. Step 5.4 writes the real
segmentation (server/perception/segmentation.py); the test then switches to it and this file is deleted.
"""
import math

import numpy as np


def cluster_labels(range_m: np.ndarray, alpha_h: np.ndarray, alpha_v: np.ndarray, use: np.ndarray, beta_deg: float = 10.0):
    """Connected components over the cells in `use`. Neighbours join when beta > beta_deg. Returns int labels, -1 = unused."""
    rows, cols = range_m.shape
    parent = list(range(rows * cols))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    thr = math.radians(beta_deg)

    def same(r1, c1, r2, c2, alpha):
        a, b = float(range_m[r1, c1]), float(range_m[r2, c2])
        d1, d2 = max(a, b), min(a, b)
        return math.atan2(d2 * math.sin(alpha), d1 - d2 * math.cos(alpha)) > thr

    for r in range(rows):
        for c in range(cols):
            if not use[r, c]:
                continue
            if c + 1 < cols and use[r, c + 1] and same(r, c, r, c + 1, alpha_h[r, c]):
                parent[find(r * cols + c)] = find(r * cols + c + 1)
            if r + 1 < rows and use[r + 1, c] and same(r, c, r + 1, c, alpha_v[r, c]):
                parent[find(r * cols + c)] = find((r + 1) * cols + c)
    labels = np.full((rows, cols), -1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            if use[r, c]:
                labels[r, c] = find(r * cols + c)
    return labels