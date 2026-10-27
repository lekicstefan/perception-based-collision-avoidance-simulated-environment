from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from server.perception.ground import GROUND, OTHER
from server.perception.range_image import LidarGeometry, RangeImage

BUILDING = 3     # label of a cell: part of a building (comes from OTHER)


@dataclass(frozen=True)
class BuildingConfig:
    line_tol_m: float        # cells of one column form a vertical line if their horizontal distances from the sensor are within this
    line_min_points: int     # a vertical line needs at least this many cells
    link_gap_m: float        # two vertical lines in neighbouring columns can be linked if they are at most this far apart (seen from above)
    flat_tol_m: float        # a line joins a surface only if it is within this of the straight line through the lines the surface has so far
    min_area_m2: float       # a surface (connected vertical lines) of at least this area is a building (this much or more)
    corner_angle_tol_deg: float   # two surfaces meet at a corner if they are perpendicular, give or take this
    corner_min_lines: int    # a surface needs at least this many vertical lines to take part in a corner
    side_offset_m: float     # a building group whose centre is at least this far sideways from the centre of the car is a left or right one
    remove_by_ground: bool   # switch for the last step: left and right buildings with ground beside them lose their label


class BuildingFinder:
    def __init__(self, geometry: LidarGeometry, config: BuildingConfig):
        self.config = config
        dirs = geometry.dirs_vehicle
        self.dir_x, self.dir_y, self.dir_z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        self.origin = geometry.origin
        self.links = np.zeros((0, 5))
        self.surfaces = np.zeros((0, 4))

    def mark(self, img: RangeImage, labels: np.ndarray) -> np.ndarray:
        c = self.config
        self.links = np.zeros((0, 5))
        self.surfaces = np.zeros((0, 4))
        valid = img.valid
        r = np.where(valid, img.range_m, 0.0).astype(np.float64)
        x = self.origin[0] + r * self.dir_x
        y = self.origin[1] + r * self.dir_y
        z = self.origin[2] + r * self.dir_z
        d = r * np.hypot(self.dir_x, self.dir_y)         # horizontal distance from the sensor
        other = valid & (labels == OTHER)
        rows, cols = other.shape

        # 1. vertical lines. In every column the cells are sorted by distance and cut into groups that are at most line_tol_m wide.
        key = np.where(other, d, np.inf)
        order = np.argsort(key, axis=0, kind="stable")
        d_sorted = np.take_along_axis(key, order, axis=0)
        group = np.full((rows, cols), -1)
        current = np.full(cols, -1)
        start = np.full(cols, -np.inf)
        for k in range(rows):
            has = np.isfinite(d_sorted[k])
            new = has & (d_sorted[k] - start > c.line_tol_m)
            current = current + new
            start = np.where(new, d_sorted[k], start)
            group[k] = np.where(has, current, -1)
        col_of, k_of = np.nonzero(np.isfinite(d_sorted).T)                 # the cells, column by column, nearest first
        if len(col_of) == 0:
            return labels
        row_of = order[k_of, col_of]
        group_of = group[k_of, col_of]
        first = np.concatenate([[True], (col_of[1:] != col_of[:-1]) | (group_of[1:] != group_of[:-1])])
        starts = np.flatnonzero(first)
        line_of_cell = np.cumsum(first) - 1                                # which group every cell is in
        count = np.diff(np.concatenate([starts, [len(col_of)]]))
        zc, xc, yc = z[row_of, col_of], x[row_of, col_of], y[row_of, col_of]
        keep = count >= c.line_min_points
        line_id = np.where(keep, np.cumsum(keep) - 1, -1)
        if not keep.any():
            return labels
        low = np.minimum.reduceat(zc, starts)[keep]
        high = np.maximum.reduceat(zc, starts)[keep]
        lx = (np.add.reduceat(xc, starts) / count)[keep]
        ly = (np.add.reduceat(yc, starts) / count)[keep]
        lcol = col_of[starts][keep]
        n = len(lcol)

        # 2. surfaces. Candidate links between a line and the lines of the next column; each line keeps its closest link
        # forward and backward, and a link counts if it is the closest one for both of its lines.
        per_col = np.bincount(lcol, minlength=cols + 1)
        first_in_col = np.cumsum(per_col) - per_col
        next_n = per_col[lcol + 1]
        a = np.repeat(np.arange(n), next_n)
        b = first_in_col[lcol[a] + 1] + np.arange(len(a)) - np.repeat(np.cumsum(next_n) - next_n, next_n)
        gap = np.hypot(lx[a] - lx[b], ly[a] - ly[b])
        ok = (np.minimum(high[a], high[b]) > np.maximum(low[a], low[b])) & (gap <= c.link_gap_m)
        a, b, gap = a[ok], b[ok], gap[ok]
        closest = np.argsort(gap, kind="stable")
        best_next = np.zeros(len(a), dtype=bool)
        best_next[closest[np.unique(a[closest], return_index=True)[1]]] = True
        best_prev = np.zeros(len(a), dtype=bool)
        best_prev[closest[np.unique(b[closest], return_index=True)[1]]] = True
        mutual = best_next & best_prev
        a, b, gap = a[mutual], b[mutual], gap[mutual]
        # follow every chain of linked lines and cut it into flat surfaces
        successor = np.full(n, -1)
        successor[a] = b
        has_previous = np.zeros(n, dtype=bool)
        has_previous[b] = True
        xs, ys, following = lx.tolist(), ly.tolist(), successor.tolist()
        surface = np.full(n, -1)
        n_surface = 0
        for head in np.flatnonzero(~has_previous).tolist():
            i, m = head, 0
            sx = sy = sxx = sxy = syy = 0.0                       # sums over the lines of the surface so far
            while i >= 0:
                px, py = xs[i], ys[i]
                if m >= 3:
                    mx, my = sx / m, sy / m
                    angle = 0.5 * math.atan2(2.0 * (sxy / m - mx * my), (sxx / m - mx * mx) - (syy / m - my * my))
                    if abs(-math.sin(angle) * (px - mx) + math.cos(angle) * (py - my)) > c.flat_tol_m:
                        n_surface += 1
                        m = 0
                        sx = sy = sxx = sxy = syy = 0.0
                surface[i] = n_surface
                m += 1
                sx, sy, sxx, sxy, syy = sx + px, sy + py, sxx + px * px, sxy + px * py, syy + py * py
                i = following[i]
            n_surface += 1
        inside = surface[a] == surface[b]                        # a link between two surfaces is a corner: not part of either
        corner_a, corner_b = a[~inside], b[~inside]
        a, b, gap = a[inside], b[inside], gap[inside]
        area = np.bincount(surface[a], weights=gap * 0.5 * ((high - low)[a] + (high - low)[b]), minlength=n_surface)
        linked = np.bincount(surface[a], minlength=n_surface) > 0
        accepted = area >= c.min_area_m2

        # 3. buildings: by area, then by corner inheritance
        n_lines = np.bincount(surface, minlength=n_surface)
        mean_x, mean_y = np.bincount(surface, weights=lx, minlength=n_surface) / n_lines, np.bincount(surface, weights=ly, minlength=n_surface) / n_lines
        cov_xx = np.bincount(surface, weights=lx * lx, minlength=n_surface) / n_lines - mean_x * mean_x
        cov_yy = np.bincount(surface, weights=ly * ly, minlength=n_surface) / n_lines - mean_y * mean_y
        cov_xy = np.bincount(surface, weights=lx * ly, minlength=n_surface) / n_lines - mean_x * mean_y
        theta = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)                    # direction of every surface
        sa, sb = surface[corner_a], surface[corner_b]
        turn = np.abs((theta[sa] - theta[sb]) % np.pi - np.pi / 2)
        corner = (n_lines[sa] >= c.corner_min_lines) & (n_lines[sb] >= c.corner_min_lines) & (turn <= np.radians(c.corner_angle_tol_deg))
        n_group, group_of = connected_components(csr_matrix((np.ones(int(corner.sum())), (sa[corner], sb[corner])), shape=(n_surface, n_surface)), directed=False)
        group_has_area = np.bincount(group_of, weights=accepted.astype(float), minlength=n_group) > 0
        building = group_has_area[group_of]                                       # per surface

        # 4. ground check
        cell_line = line_id[line_of_cell]
        used = cell_line >= 0
        cell_surface = np.full(len(col_of), -1)
        cell_surface[used] = surface[cell_line[used]]
        in_building = np.zeros(len(col_of), dtype=bool)
        in_building[used] = building[cell_surface[used]]
        group_stays = np.ones(n_group, dtype=bool)
        if c.remove_by_ground:
            ground = labels == GROUND
            cells = np.flatnonzero(in_building)
            cell_group = group_of[cell_surface[cells]]
            for g in np.unique(cell_group):
                mine = cells[cell_group == g]
                centre = 0.5 * (yc[mine].min() + yc[mine].max())
                bottom = int(row_of[mine].max())
                if abs(centre) < c.side_offset_m or bottom == 0:
                    continue
                y_bottom = yc[mine[row_of[mine] == bottom]]
                beyond = y[bottom] > y_bottom.max() if centre > 0 else y[bottom] < y_bottom.min()
                beside = np.flatnonzero(ground[bottom] & beyond)
                if ground[bottom - 1, beside].any():
                    group_stays[g] = False

        # 5. labels and report data
        kept = np.flatnonzero(in_building)
        in_building[kept] = group_stays[group_of[cell_surface[kept]]]
        labels[row_of[in_building], col_of[in_building]] = BUILDING
        status = np.where(building, np.where(group_stays[group_of], 1, 2), 0)
        la, lb = np.concatenate([a, corner_a[corner]]), np.concatenate([b, corner_b[corner]])
        self.links = np.column_stack([lx[la], ly[la], lx[lb], ly[lb], status[surface[la]]]).astype(float)
        self.surfaces = np.column_stack([mean_x, mean_y, area, status])[linked].astype(float)
        return labels