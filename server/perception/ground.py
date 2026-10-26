from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import percentile_filter

from server.perception.range_image import LidarGeometry, RangeImage

NONE, GROUND, OTHER = 0, 1, 2     # label of a cell: no return, ground (road, curb, sidewalk), anything else
# why a cell is OTHER (kept for debugging)
NOT_OTHER, TOO_HIGH, TOO_LOW, BEHIND, STEP_TOO_FAR, SECOND_STEP, CURB_REJECTED, ROW_OUTLIER, FOOT = range(9)
REASON_NAMES = ["-", "above the predicted ground", "below the predicted ground", "nearer than the last ground point",
                "small step but the last ground point is too far", "second step in a row", "curb step not confirmed by the next point",
                "sticks out from its neighbours in the row", "lowest cell under an object, on its vertical face"]


@dataclass(frozen=True)
class GroundConfig:
    noise_tol_m: float        # a point this close to the predicted ground height is ground (range noise, small bumps)
    slope_error_deg: float    # how wrong the slope estimate may be at the start of a ray, before any ground is seen
    curvature: float          # 1/m, how fast the ground may bend up or down (adds 0.5 * curvature * gap^2 to the tolerance)
    start_tol_m: float        # the first ground point of a ray may be this far from the plane of the car's wheels
    curb_height_m: float      # a step up of at most this much, followed by ground at the new height, is a curb
    curb_reach_m: float       # a curb step is only recognised between two returns at most this far apart (horizontally)
    close_reach_m: float      # between two returns at most this far apart, a gentle slope is ground (a kink in the road)
    close_slope_deg: float    # "gentle" = at most this steep
    slope_max_deg: float      # steepest slope the estimate may follow
    slope_memory_m: float     # the slope estimate averages over about this much ground
    behind_tol_m: float       # a ground point may be this much nearer than the last ground point
    min_move_m: float         # the slope estimate is only updated by points at least this far beyond the last ground point
    row_window: int           # columns to each side used to compare a ground point with its neighbours in the same row
    row_percentile: int       # the neighbours are compared at this percentile of their height error
    row_outlier_m: float      # a ground point this far above its neighbours in the row is not ground
    foot_slope_deg: float     # a ground cell right under an "other" cell, on a line at least this steep, is part of the object


def level_frame(points: np.ndarray, pitch: float):
    """Vehicle frame points -> (forward distance along the level plane, y, height above the plane through the vehicle origin)."""
    s, c = np.sin(pitch), np.cos(pitch)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return c * x - s * z, y, s * x + c * z


class GroundRemover:
    def __init__(self, geometry: LidarGeometry, config: GroundConfig):
        self.geometry = geometry
        self.config = config
        dirs = geometry.dirs_vehicle
        self.hdir = np.hypot(dirs[..., 0], dirs[..., 1])      # horizontal distance per metre of range
        self.vdir = dirs[..., 2]
        self.sensor_z = float(geometry.origin[2])             # height of the sensor above the plane of the car's wheels
        self.reasons = np.zeros((geometry.rows, geometry.cols), dtype=np.uint8)   # of the last classify call

    def classify(self, img: RangeImage) -> np.ndarray:
        """Label of every cell (NONE, GROUND, OTHER), shape (rows, cols).

        Every column of the range image is one ray direction seen from above, so it is followed from the lowest beam up
        as a profile of horizontal distance d and height z. Ground is what continues the ground seen so far."""
        c = self.config
        valid = img.valid
        r = np.where(valid, img.range_m, 0.0).astype(np.float64)
        d = r * self.hdir
        z = self.sensor_z + r * self.vdir
        rows, cols = r.shape
        label = np.zeros((rows, cols), dtype=np.uint8)
        reason = np.zeros((rows, cols), dtype=np.uint8)
        dg = np.zeros(cols)                    # last ground point of each column: distance, height
        zg = np.zeros(cols)
        slope = np.zeros(cols)
        known = np.zeros(cols, dtype=bool)     # the column has a slope estimate
        stepped = np.zeros(cols, dtype=bool)   # the last accepted point was a curb step
        pend = np.zeros(cols, dtype=bool)      # that step is not confirmed yet
        pend_row = np.zeros(cols, dtype=int)
        saved = np.zeros((4, cols))            # state before the pending step: dg, zg, slope, known
        tan_max = np.tan(np.radians(c.slope_max_deg))
        tan_err = np.tan(np.radians(c.slope_error_deg))
        tan_close = np.tan(np.radians(c.close_slope_deg))
        cols_idx = np.arange(cols)
        for i in range(rows - 1, -1, -1):
            v = valid[i]
            if not v.any():
                continue
            di, zi = d[i], z[i]

            def judge():
                delta = np.maximum(di - dg, 0.0)
                near = (delta <= c.close_reach_m) & (np.abs(zi - zg) <= tan_close * delta + c.noise_tol_m)
                e = zi - (zg + slope * delta)                              # height above the predicted ground
                unc = np.where(known, tan_err * c.slope_memory_m / (c.slope_memory_m + dg), tan_err)
                tol = c.noise_tol_m + unc * delta + 0.5 * c.curvature * delta * delta
                tol = np.where(dg > 0, tol, np.maximum(tol, c.start_tol_m))
                return delta, e, tol, di >= dg - c.behind_tol_m, near

            delta, e, tol, ahead, near = judge()
            # a curb step counts only if the ground goes on at the new height: a point that does not fit takes the step back
            bad = pend & v & ~(ahead & ((np.abs(e) <= tol) | near))
            if bad.any():
                label[pend_row[bad], cols_idx[bad]] = OTHER
                reason[pend_row[bad], cols_idx[bad]] = CURB_REJECTED
                dg[bad], zg[bad], slope[bad], known[bad] = saved[0][bad], saved[1][bad], saved[2][bad], saved[3][bad] > 0
                stepped[bad] = False
                delta, e, tol, ahead, near = judge()
            pend &= ~v
            ok = v & ahead & ((np.abs(e) <= tol) | near)
            small = v & ahead & ~ok & (e > 0) & (e <= c.curb_height_m)
            step = small & (delta <= c.curb_reach_m) & ~stepped
            reason[i] = np.where(v & ~ok & ~step, np.where(~ahead, BEHIND, np.where(small, np.where(stepped, SECOND_STEP, STEP_TOO_FAR),
                                                                                  np.where(e > 0, TOO_HIGH, TOO_LOW))), 0)
            if ok.sum() > 2 * c.row_window:
                # a ground point that stands out from the ground points beside it in the same row is not ground
                fill = np.where(ok, zi, np.inf)
                low = percentile_filter(fill, c.row_percentile, size=2 * c.row_window + 1, mode="constant", cval=np.inf)
                outlier = ok & (zi - low > c.row_outlier_m)
                reason[i][outlier] = ROW_OUTLIER
                ok &= ~outlier
            label[i][v] = OTHER
            label[i][ok | step] = GROUND
            moved = ok & (delta > c.min_move_m)
            seg = (zi - zg) / np.maximum(delta, 1e-6)
            gain = delta / (delta + c.slope_memory_m)
            new = np.where(known, slope + gain * (seg - slope), seg)
            saved[:, step] = np.stack([dg, zg, slope, known.astype(float)])[:, step]
            pend_row[step] = i
            pend |= step
            slope = np.where(moved, np.clip(new, -tan_max, tan_max), slope)
            known |= moved
            take = ok | step
            dg = np.where(take, di, dg)
            zg = np.where(take, zi, zg)
            stepped = np.where(step, True, np.where(ok, False, stepped))
        # the lowest cell that hits an object looks like gentle ground from the cell below it, but the cell above it is
        # on a near-vertical line with it, so walking down each column such a cell is given to the object
        tan_foot = np.tan(np.radians(c.foot_slope_deg))
        above_d, above_z, above_other = np.zeros(cols), np.zeros(cols), np.zeros(cols, dtype=bool)
        for i in range(rows):
            v = valid[i]
            rise = above_z - z[i]
            foot = v & (label[i] == GROUND) & above_other & (rise > c.noise_tol_m) & (rise >= tan_foot * np.abs(above_d - d[i]))
            label[i][foot] = OTHER
            reason[i][foot] = FOOT
            above_d, above_z, above_other = np.where(v, d[i], above_d), np.where(v, z[i], above_z), np.where(v, label[i] == OTHER, above_other)
        self.reasons = reason
        return label