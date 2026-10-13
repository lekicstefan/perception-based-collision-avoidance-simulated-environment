"""Data association (step 6.4): which measurement belongs to which track.

Per scan there are N tracks (each with a predicted position and covariance) and M measurements (position z, covariance R, the
capture time of the scan). The squared Mahalanobis distance of measurement j from track i,

    d2_ij = y^T S^-1 y,   y = z_j - H x_i(t),   S = H P_i(t) H^T + R_j,

says how many standard deviations of the combined uncertainty (track prediction plus measurement) the two are apart; for a
correct pair it is chi-square distributed with 2 degrees of freedom (mean 2). Two steps:

  gate     a pair with d2 above `gate_chi2` is not possible (9.21 keeps 99 % of the correct pairs, 13.8 keeps 99.9 %)
  assign   the Hungarian algorithm finds the one-to-one assignment with the smallest sum of d2 among the possible pairs
           (global nearest neighbour). A track never gets two measurements and a measurement never two tracks; what is left
           over is `unmatched_tracks` (they coast, step 6.5) and `unmatched_measurements` (they may start new tracks).

The Hungarian algorithm is written out here (O(n^3), a few dozen lines) so that the processor needs nothing but numpy.
`associate` only reads the filters (`gate_distance2` does not change them); updating them is the caller's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AssociationConfig:
    gate_chi2: float = 18.4             # squared Mahalanobis gate, chi-square 2 dof: 5.99 = 95 %, 9.21 = 99 %, 13.8 = 99.9 %, 18.4 = 99.99 % (tuned in 6.9, was 9.21)


@dataclass
class Association:
    matches: list = field(default_factory=list)                  # [(track index, measurement index)]
    unmatched_tracks: list = field(default_factory=list)
    unmatched_measurements: list = field(default_factory=list)
    d2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))     # (tracks, measurements) squared Mahalanobis distances


def hungarian(cost) -> list:
    """Minimum-cost assignment of an (n, m) cost matrix: a list of (row, column), min(n, m) pairs, every row and column at
    most once. The shortest augmenting path algorithm with row and column potentials."""
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return []
    n, m = a.shape
    if n > m:
        return sorted((c, r) for r, c in hungarian(a.T))
    inf = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


def mahalanobis_matrix(filters, measurements, pair_measurement=None) -> np.ndarray:
    """(tracks, measurements) squared Mahalanobis distances. A measurement is (z, R, t) as `measurement_of` returns it; every
    track is predicted to the time of the measurement. With `pair_measurement(i, j)` the measurement of cluster j as seen by
    track i is used instead (the size memory of a track changes its centre, step 6.11)."""
    d2 = np.empty((len(filters), len(measurements)))
    for i, f in enumerate(filters):
        for j in range(len(measurements)):
            z, r, t = measurements[j] if pair_measurement is None else pair_measurement(i, j)
            d2[i, j] = f.gate_distance2(z, r, t)
    return d2


def associate(filters, measurements, config: AssociationConfig = AssociationConfig(), pair_measurement=None) -> Association:
    """Gate and assign. `filters`: ConstantVelocityFilter objects; `measurements`: (z, R, t) tuples; `pair_measurement`: optional
    function (track index, measurement index) -> (z, R, t), see `mahalanobis_matrix`."""
    n, m = len(filters), len(measurements)
    d2 = mahalanobis_matrix(filters, measurements, pair_measurement)
    if n == 0 or m == 0:
        return Association([], list(range(n)), list(range(m)), d2)
    possible = d2 <= config.gate_chi2
    big = (config.gate_chi2 + 1.0) * (max(n, m) + 1)             # a gated pair costs more than any set of possible pairs
    pairs = [(i, j) for i, j in hungarian(np.where(possible, d2, big)) if possible[i, j]]
    used_t, used_m = {i for i, _ in pairs}, {j for _, j in pairs}
    return Association(pairs, [i for i in range(n) if i not in used_t], [j for j in range(m) if j not in used_m], d2)