from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.perception.ground import GROUND, OTHER
from server.perception.range_image import LidarGeometry, RangeImage

BUILDING = 3     # label of a cell: part of a building (comes from OTHER)


@dataclass(frozen=True)
class BuildingConfig:
    line_tol_m: float        # in one column, cells whose horizontal distance differs by at most this are on the same vertical line
    min_height_m: float      # a vertical line this tall (highest cell minus lowest cell on it) is a wall point
    min_gap_m: float         # a wall point must be at least this far (sideways) outside the edge of the road to count
    hough_angle_deg: float   # angle step of the Hough transform
    hough_rho_m: float       # distance step of the Hough transform, also how close to a line a point must be to belong to it
    min_points: int          # a line needs at least this many wall points
    min_length_m: float      # a line needs its wall points to span at least this length
    max_lines: int           # at most this many strongest lines are taken from the Hough transform of one scan


class BuildingFinder:
    def __init__(self, geometry: LidarGeometry, config: BuildingConfig):
        self.config = config
        self.rows, self.cols = geometry.rows, geometry.cols
        dirs = geometry.dirs_vehicle
        self.dir_x, self.dir_y, self.dir_z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        self.origin = geometry.origin
        self.angles = np.radians(np.arange(0.0, 180.0, config.hough_angle_deg))
        self.rho_n = int(np.ceil((geometry.max_range_m + 2.0) / config.hough_rho_m))
        self.higher = np.tril(np.ones((self.rows, self.rows), dtype=bool), -1)      # [i, k] is true when beam k is above beam i
        self.lines: list = []        # the lines found in the last scan, as (x0, y0, x1, y1) in the vehicle frame, for the report

    def mark(self, img: RangeImage, labels: np.ndarray) -> np.ndarray:
        """Turns OTHER cells into BUILDING cells, in place, and returns the labels.

        1. Wall points: a vertical line (cells of one column at the same horizontal distance) that is tall enough is a wall
           point, seen from above, but only outside the road: sideways at least min_gap_m beyond the outermost GROUND cell
           of the rows that see the ground.
        2. Hough transform of the wall points in the top-down view (x forward, y left): the strongest line is found, refitted
           to the points close to it, and kept if it has enough points and is long enough. Its points are removed and the
           next strongest line is looked for.
        3. Every OTHER cell close to a kept line, between the ends of its points, is a BUILDING. Cells in front of the wall
           are not close to the line and stay OTHER."""
        c = self.config
        self.lines = []
        valid = img.valid
        r = np.where(valid, img.range_m, 0.0).astype(np.float64)
        x = self.origin[0] + r * self.dir_x
        y = self.origin[1] + r * self.dir_y
        z = self.origin[2] + r * self.dir_z
        d = r * np.hypot(self.dir_x, self.dir_y)         # horizontal distance from the sensor
        other = valid & (labels == OTHER)
        points = self._wall_points(x, y, z, d, other, labels == GROUND)
        for nx, ny, offset, centre, direction, lo, hi in self._hough(points):
            along = (x - centre[0]) * direction[0] + (y - centre[1]) * direction[1]
            near = other & (np.abs(x * nx + y * ny - offset) <= c.hough_rho_m) & (along >= lo - c.hough_rho_m) & (along <= hi + c.hough_rho_m)
            labels[near] = BUILDING
            self.lines.append((*(centre + lo * direction), *(centre + hi * direction)))
        return labels

    def _wall_points(self, x, y, z, d, other, ground) -> np.ndarray:
        c = self.config
        # close[i, k, col]: cells i and k of a column are both OTHER and on the same vertical line
        close = other[:, None, :] & other[None, :, :] & (np.abs(d[:, None, :] - d[None, :, :]) <= c.line_tol_m)
        top = np.where(close, z[None, :, :], -np.inf).max(axis=1)
        bottom = np.where(close, z[None, :, :], np.inf).min(axis=1)
        # one point per vertical line: the cell of its highest beam
        wall = other & (top - bottom >= c.min_height_m) & ~(close & self.higher[:, :, None]).any(axis=1)
        sees = np.flatnonzero(ground.any(axis=1))
        if len(sees) == 0 or not wall.any():
            return np.zeros((0, 2))
        # edges of the road: in every row that sees the ground, its leftmost and rightmost ground cell
        left = np.where(ground, y, -np.inf).argmax(axis=1)[sees]
        right = np.where(ground, y, np.inf).argmin(axis=1)[sees]
        left_x, left_y, right_x, right_y = x[sees, left], y[sees, left], x[sees, right], y[sees, right]
        ol, orr = np.argsort(left_x), np.argsort(right_x)
        px, py = x[wall], y[wall]
        edge_left = np.interp(px, left_x[ol], left_y[ol])
        edge_right = np.interp(px, right_x[orr], right_y[orr])
        outside = (py > edge_left + c.min_gap_m) | (py < edge_right - c.min_gap_m)
        return np.column_stack([px[outside], py[outside]])

    def _hough(self, points: np.ndarray) -> list:
        """Lines as (normal x, normal y, offset, centre, direction, start, end), start and end along the direction from the centre."""
        c = self.config
        n_angle = len(self.angles)
        cos_t, sin_t = np.cos(self.angles), np.sin(self.angles)
        size = 2 * self.rho_n + 1
        remaining = np.ones(len(points), dtype=bool)
        found = []
        for _ in range(c.max_lines):
            p = points[remaining]
            if len(p) < c.min_points:
                break
            rho = p[:, :1] * cos_t + p[:, 1:] * sin_t                                 # (points, angles)
            index = np.rint(rho / c.hough_rho_m).astype(int) + self.rho_n
            votes = np.bincount((index * n_angle + np.arange(n_angle)).ravel(), minlength=size * n_angle).reshape(size, n_angle)
            smooth = votes.copy()                                                     # a line between two distance steps keeps its votes
            smooth[1:] += votes[:-1]
            smooth[:-1] += votes[1:]
            best = int(smooth.argmax())
            if smooth.flat[best] < c.min_points:
                break
            ri, ti = divmod(best, n_angle)
            normal = np.array([cos_t[ti], sin_t[ti]])
            offset = (ri - self.rho_n) * c.hough_rho_m
            for _ in range(2):                                                        # refit the line to the points close to it
                support = np.abs(p @ normal - offset) <= c.hough_rho_m
                if support.sum() < 2:
                    break
                q = p[support]
                centre = q.mean(axis=0)
                direction = np.linalg.svd(q - centre, full_matrices=False)[2][0]
                normal = np.array([-direction[1], direction[0]])
                offset = float(centre @ normal)
            support = np.abs(p @ normal - offset) <= c.hough_rho_m
            remaining[np.flatnonzero(remaining)[support]] = False
            if support.sum() < c.min_points:
                continue
            q = p[support]
            centre = q.mean(axis=0)
            direction = np.array([normal[1], -normal[0]])
            along = (q - centre) @ direction
            if along.max() - along.min() < c.min_length_m:
                continue
            found.append((normal[0], normal[1], offset, centre, direction, along.min(), along.max()))
        return found