"""Angle-based segmentation (step 5.4).

Groups the candidate cells (valid returns that are not ground, see ground.py) of a range image into clusters, one per
object, without any model of what the object is.

  1. beta criterion. For two neighbouring cells with ranges d1 (the farther) and d2, separated by the beam angle alpha,
         beta = atan2(d2 sin(alpha), d1 - d2 cos(alpha))
     If beta exceeds a threshold (default 10 degrees) the cells belong to the same object. beta is the angle at the
     farther point between the line of sight and the line to the nearer point, so a surface seen face on has a large
     beta and a depth jump has a small one, at any range, which a fixed distance threshold cannot do.
  2. Connected components over the 4-neighbourhood of the range image (the edges are computed vectorized, the components
     by scipy). Cells that are not candidates (no return, max range, ground) have no edges: they are boundaries.
  3. Secondary split test. beta can merge touching objects at similar range (a person next to a parked car). Each cluster's
     depth profile (nearest range per column) and height profile (top of the cluster per column) are checked for a step
     between neighbouring columns, and the cluster is cut there. A step must be larger than an absolute minimum and than
     2.5 times the jumps next to it (so a slanted wall or the corner of a car, whose depth changes steadily, is not cut)
     and, for the height profile, larger than 1.5 times the largest gap between the beams of the cluster at that range (the top is only sampled once per beam).
  4. Range-dependent minimum size. The minimum is a physical width and height, converted into cells at the cluster's
     (median) range: a 0.5 m wide object at 60 m is only 2 to 3 columns wide. Clusters spanning fewer columns or rows
     than that, or fewer than min_cells cells, are discarded.

Row/column conventions are those of range_image.py. Heights are z of the vehicle frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from server.perception.range_image import LidarGeometry, RangeImage


@dataclass(frozen=True)
class SegmentationConfig:
    beta_deg: float = 10.0
    min_width_m: float = 0.3         # physical minimum size of a cluster, converted to cells at its range
    min_height_m: float = 0.3
    min_cells: int = 2
    split: bool = True               # secondary split test on/off (off only to show what it does)
    split_depth_step_m: float = 0.15
    split_height_step_m: float = 0.2
    merge_caps: bool = True                    # join a flat strip on top of an object (its roof, seen from above) to it
    cap_max_rows: int = 2
    cap_wide_cells: int = 14                   # a strip this big and this wide (not a far person) may also overlap the top row
    cap_wide_cols: int = 10
    cap_max_ratio: float = 0.35                # the strip has at most this share of the object's cells
    cap_depth_m: float = 6.0                   # and lies at most this far behind the object's nearest range
    cap_height_tol_m: float = 0.15             # its height may exceed the object's top by at most this
    split_height_beam_factor: float = 0.6      # a height step must also exceed this many times the largest gap between two beams


@dataclass(frozen=True, eq=False)
class Cluster:
    label: int
    rows: np.ndarray                 # row indices of the cells
    cols: np.ndarray                 # column indices of the cells
    range_m: float                   # median range of the cells

    @property
    def n_cells(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, eq=False)
class Segmentation:
    labels: np.ndarray               # int32 (rows, cols): cluster label (0 ...), -1 = not part of a kept cluster
    clusters: list                   # list of Cluster, labels 0 ... n - 1
    n_discarded: int                 # clusters thrown away for being too small


def beta_angle(a, b, alpha):
    """beta (rad) of two ranges a, b separated by the angle alpha. Works on arrays."""
    d1, d2 = np.maximum(a, b), np.minimum(a, b)
    return np.arctan2(d2 * np.sin(alpha), d1 - d2 * np.cos(alpha))


def beta_labels(range_m: np.ndarray, alpha_h: np.ndarray, alpha_v: np.ndarray, use: np.ndarray,
                beta_deg: float = 10.0) -> np.ndarray:
    """Connected components of the cells in `use`, neighbours joined when beta > beta_deg. Labels 0 ..., -1 = unused."""
    rows, cols = range_m.shape
    r = np.where(use, range_m, 1.0).astype(np.float64)              # unused cells never get an edge, the value is a dummy
    thr = math.radians(beta_deg)
    idx = np.arange(rows * cols).reshape(rows, cols)
    ok_h = use[:, :-1] & use[:, 1:] & (beta_angle(r[:, :-1], r[:, 1:], alpha_h) > thr)
    ok_v = use[:-1] & use[1:] & (beta_angle(r[:-1], r[1:], alpha_v) > thr)
    a = np.concatenate([idx[:, :-1][ok_h], idx[:-1][ok_v]])
    b = np.concatenate([idx[:, 1:][ok_h], idx[1:][ok_v]])
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(rows * cols, rows * cols))
    _, comp = connected_components(graph, directed=False)
    comp = comp.reshape(rows, cols)
    labels = np.full((rows, cols), -1, dtype=np.int32)
    _, labels[use] = np.unique(comp[use], return_inverse=True)      # contiguous labels, only for the used cells
    return labels


class Segmenter:
    def __init__(self, geometry: LidarGeometry, config: SegmentationConfig | None = None):
        self.geometry = geometry
        self.config = config or SegmentationConfig()
        self.az_step = float(geometry.alpha_h.mean())                      # rad between neighbouring columns
        rs = geometry.alpha_v.mean(axis=1)                                 # rad between a beam and the one below it
        self.row_step = np.append(rs, rs[-1]) if len(rs) else np.array([self.az_step])

    # ------------------------------------------------------------------
    def segment(self, img: RangeImage, candidates: np.ndarray) -> Segmentation:
        cfg = self.config
        rng = np.where(candidates, img.range_m, np.nan)
        labels = beta_labels(img.range_m, self.geometry.alpha_h, self.geometry.alpha_v, candidates, cfg.beta_deg)
        z = self.geometry.points_vehicle(img)[..., 2]
        groups = self._groups(labels)
        if cfg.split:
            groups = [g for rows, cols in groups for g in self._split(rows, cols, rng, z)]
        if cfg.merge_caps:
            groups = self._merge_caps(groups, rng, z)
        out = np.full(img.shape, -1, dtype=np.int32)
        clusters, discarded = [], 0
        for rows, cols in groups:
            if not self._big_enough(rows, cols, rng):
                discarded += 1
                continue
            k = len(clusters)
            out[rows, cols] = k
            clusters.append(Cluster(k, rows, cols, float(np.median(rng[rows, cols]))))
        return Segmentation(out, clusters, discarded)

    @staticmethod
    def _groups(labels: np.ndarray):
        r, c = np.nonzero(labels >= 0)
        if len(r) == 0:
            return []
        lab = labels[r, c]
        order = np.argsort(lab, kind="stable")
        r, c, lab = r[order], c[order], lab[order]
        cuts = np.flatnonzero(np.diff(lab)) + 1
        return list(zip(np.split(r, cuts), np.split(c, cuts)))

    def _merge_caps(self, groups, rng, z):
        """A car seen from above has its roof in one or two rows that lie far behind its front (the beam grazes the roof),
        so the angle test cuts them off. A strip of at most cap_max_rows rows right above a bigger group, inside its
        columns, whose height matches the group's top and which is not too far behind it, is joined to that group.
        A person's head behind a car is taller than the car's top and keeps its own cluster."""
        cfg = self.config
        info = [(rows, cols, rng[rows, cols], z[rows, cols]) for rows, cols in groups]
        target = list(range(len(groups)))
        for a, (ra, ca, da, za) in enumerate(info):
            if np.ptp(ra) + 1 > cfg.cap_max_rows:
                continue
            wide = len(ra) >= cfg.cap_wide_cells and np.ptp(ca) + 1 >= cfg.cap_wide_cols
            best = None
            for b, (rb, cb, db, zb) in enumerate(info):
                if b == a or len(ra) > cfg.cap_max_ratio * len(rb):
                    continue
                if wide:                                                                      # on top of it, may overlap its top row
                    if not (ra.max() >= rb.min() - 2 and ra.min() <= rb.min() + 1):
                        continue
                elif not (rb.min() - 2 <= ra.max() < rb.min()):                               # small: entirely just above it
                    continue
                    continue
                if (ca < cb.min() - 1).any() or (ca > cb.max() + 1).any():                  # inside its columns
                    continue
                near = float(np.nanmin(db))
                if not (near - 0.5 <= float(np.nanmedian(da)) <= near + cfg.cap_depth_m):
                    continue
                if float(np.nanmax(za)) > float(np.nanmax(zb)) + cfg.cap_height_tol_m:
                    continue
                if best is None or len(rb) > len(info[best][0]):
                    best = b
            if best is not None:
                target[a] = best
        merged = {}
        for k, (rows, cols) in enumerate(groups):
            t = target[k]
            merged.setdefault(t, []).append((rows, cols))
        return [(np.concatenate([r for r, _ in v]), np.concatenate([c for _, c in v])) for v in merged.values()]

    def _split(self, rows, cols, rng, z):
        """Cut a cluster at steps in its depth or height profile. Returns a list of (rows, cols)."""
        cfg = self.config
        ucols = np.unique(cols)
        if len(ucols) < 2:
            return [(rows, cols)]
        pos = np.searchsorted(ucols, cols)
        depth = np.full(len(ucols), np.inf)
        top = np.full(len(ucols), -np.inf)
        np.minimum.at(depth, pos, rng[rows, cols])
        np.maximum.at(top, pos, z[rows, cols])
        adjacent = np.diff(ucols) == 1                                    # a gap in the columns is not a step
        if not adjacent.any():
            return [(rows, cols)]
        r_med = float(np.median(rng[rows, cols]))
        vcell = r_med * float(self.row_step[max(rows.min() - 1, 0):rows.max() + 1].max())   # largest gap between beams
        dd, dz = np.abs(np.diff(depth)), np.abs(np.diff(top))
        cut = adjacent & (self._is_step(dd, adjacent, cfg.split_depth_step_m) |
                          self._is_step(dz, adjacent, max(cfg.split_height_step_m, cfg.split_height_beam_factor * vcell)))
        if not cut.any():
            return [(rows, cols)]
        part = np.searchsorted(ucols[:-1][cut], cols, side="left")        # which side of each cut the column is on
        return [(rows[part == p], cols[part == p]) for p in np.unique(part)]

    @staticmethod
    def _is_step(delta: np.ndarray, adjacent: np.ndarray, minimum: float) -> np.ndarray:
        """Which jumps of a profile are steps: larger than `minimum` and than 2.5 times the jumps next to them. A slanted
        or L-shaped surface changes steadily, so its jumps are as large as their neighbours' and none is a step."""
        d = np.where(adjacent, delta, 0.0)
        left = np.concatenate([[0.0], d[:-1]])
        right = np.concatenate([d[1:], [0.0]])
        return d > np.maximum(minimum, 2.5 * np.maximum(left, right))

    def _big_enough(self, rows, cols, rng) -> bool:
        cfg = self.config
        if len(rows) < cfg.min_cells:
            return False
        r_med = float(np.median(rng[rows, cols]))
        min_cols = max(1, int(cfg.min_width_m / (r_med * self.az_step)))
        min_rows = max(1, int(cfg.min_height_m / (r_med * float(self.row_step[rows].mean()))))
        return (cols.max() - cols.min() + 1) >= min_cols and (rows.max() - rows.min() + 1) >= min_rows