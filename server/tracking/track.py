"""One track and its lifecycle (step 6.5).

                 3 consecutive hits                      miss
    TENTATIVE ------------------------> CONFIRMED -----------------> COASTING
        |  (more than `tentative_max_misses`    ^  <---- hit ----------   |
        |   misses in a row: deleted)           |                         | K = `max_misses` misses in a row: deleted
        v                                       +-------------------------+
     deleted

A new track starts tentative with the position of its first measurement and a large velocity uncertainty (the filter does
that). Only confirmed and coasting tracks are reported to the risk stage (confirmation keeps single-scan noise from braking
the car). A coasting track keeps predicting with its velocity; its covariance grows with every miss, so the gate grows too.
How long a coasting track lives: K misses, more (`coast_bonus_misses`) for an old track with a high hit ratio, and as long
as a camera bearing keeps arriving (`camera_keep_s`) up to `hard_miss_factor` x K misses.
The track also remembers the largest extent it has been seen with (`extent`, step 6.11): the tracked centre then does not move
when only a part of a big object is in view.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from server.tracking.kalman import ConstantVelocityFilter
from server.tracking.measurement import WorldCluster
from server.tracking.raw_baseline import raw_velocity


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


@dataclass(frozen=True)
class TrackConfig:
    confirm_hits: int = 3               # M: consecutive hits that confirm a track
    max_misses: int = 5                 # K: consecutive misses after which a confirmed track is deleted
    tentative_max_misses: int = 2       # consecutive misses after which a tentative track is deleted
    merge_scans: int = 3                # scans in which a track must be explained by another track's cluster to be merged into it
    existence_window: int = 50          # scans over which the hit ratio (existence) is taken
    coast_bonus_misses: int = 5         # a long-lived, reliably seen track survives this many more misses ...
    coast_bonus_min_ratio: float = 0.8  # ... when its hit ratio is at least this ...
    coast_bonus_min_age: int = 20       # ... and it is at least this many scans old (0 bonus misses: the plain rule of K)
    camera_keep_s: float = 1.0          # a coasting track with a camera bearing this recently is not deleted for its misses ...
    hard_miss_factor: int = 3           # ... until it has missed this many times K
    duplicate_scans: int = 3            # two tracks at the same place (below) in this many scans, not necessarily in a row, are one (0: off)
    duplicate_max_dist_m: float = 0.6   # ... same place: positions closer than this ...
    duplicate_chi2: float = 9.21        # ... and within this chi-square (2 dof) of their position covariances ...
    duplicate_max_dv_mps: float = 1.0   # ... and velocities closer than this


def surface_offsets(w: WorldCluster, position) -> np.ndarray:
    """(5, 2): the four corners of the visible box of a cluster and its nearest point, as offsets from the tracked point `position`.
    The risk stage moves these points with the track (also while it coasts) to get the surface that can be hit, which is not the
    tracked centre (that is biased along the line of sight, phase 6)."""
    pts = np.vstack([np.asarray(w.corners_xy, dtype=float).reshape(-1, 2), np.asarray(w.nearest_xy, dtype=float)[None]])
    return pts - np.asarray(position, dtype=float)


class Track:
    def __init__(self, track_id: int, flt: ConstantVelocityFilter, w: WorldCluster, config: TrackConfig = TrackConfig()):
        self.id = track_id
        self.filter = flt
        self.config = config
        self.state = TrackState.TENTATIVE
        self.born_t = flt.t
        self.hits = 1                   # measurements assigned in total
        self.consecutive_hits = 1
        self.misses = 0                 # consecutive scans without a measurement
        self.age_scans = 1
        self.history = deque([True], maxlen=config.existence_window)       # LiDAR hit (True) or miss per scan
        self.raw_velocity = None                                            # two-frame difference of the last two measurements (baseline)
        self._prev_measurement = (flt.position.copy(), flt.t)
        self.static = False                                                 # motion flag (outputs.py)
        self.static_count = 0
        self.camera_hits = 0            # camera bearing updates applied (step 6.6); evidence of existence for the risk stage
        self.last_camera_t = None
        self.shared = 0                 # consecutive scans in which another track's cluster also fits this track (merge test)
        self.last_hit_t = flt.t
        self.last_cluster = w
        self.surface_offsets = surface_offsets(w, flt.position)      # the visible surface relative to the tracked point (see above)
        self.max_width_m = w.width_m
        self.max_depth_m = w.depth_m
        self.max_height_m = w.height_m
        self.extent = None              # remembered size per world axis (extent.py, set by the tracker)
        self.eff_extent = None          # extent the tracked point is defined with now: the memory, or the visible hull when larger
        self.recent_hulls = []          # hull sizes of the latest clusters (extent.grow: persistence of a larger size)

    # ---- events -----------------------------------------------------------------------------------------------------
    def hit(self, w: WorldCluster, z, r, t: float) -> bool:
        """A measurement was assigned. Returns True when this hit confirmed the track."""
        self.filter.update(z, r, t)
        self.raw_velocity = raw_velocity(*self._prev_measurement, z, t)
        self._prev_measurement = (np.asarray(z, dtype=float).copy(), t)
        self.hits += 1
        self.consecutive_hits += 1
        self.misses = 0
        self.shared = 0
        self.age_scans += 1
        self.history.append(True)
        self.last_hit_t = t
        self.last_cluster = w
        self.surface_offsets = surface_offsets(w, self.filter.position)
        self.max_width_m = max(self.max_width_m, w.width_m)
        self.max_depth_m = max(self.max_depth_m, w.depth_m)
        self.max_height_m = max(self.max_height_m, w.height_m)
        was = self.state
        if self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.TENTATIVE and self.consecutive_hits >= self.config.confirm_hits:
            self.state = TrackState.CONFIRMED
        return was == TrackState.TENTATIVE and self.state == TrackState.CONFIRMED

    def rereference(self, delta) -> None:
        """The tracked point is defined differently (the remembered extent grew, extent.py): move the track by delta (2,) so
        that the filter does not read it as motion. The previous measurement of the raw baseline moves with it."""
        delta = np.asarray(delta, dtype=float)
        self.filter.x[:2] += delta
        if getattr(self, "surface_offsets", None) is not None:
            self.surface_offsets = self.surface_offsets - delta            # the surface stays where it is, only the point moved
        pos, t = self._prev_measurement
        self._prev_measurement = (pos + delta, t)

    def camera_hit(self, t: float) -> None:
        self.camera_hits += 1
        self.last_camera_t = t

    def miss(self, t: float) -> bool:
        """No measurement in this scan. Returns True when the track is to be deleted."""
        if t > self.filter.t:                                       # a camera update may already have moved the track past t
            self.filter.predict(t)
        self.consecutive_hits = 0
        self.misses += 1
        self.age_scans += 1
        self.history.append(False)
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.COASTING
        if self.state == TrackState.TENTATIVE:
            return self.misses > self.config.tentative_max_misses
        cfg = self.config
        limit = cfg.max_misses
        if self.age_scans >= cfg.coast_bonus_min_age and self.reliability >= cfg.coast_bonus_min_ratio:
            limit += cfg.coast_bonus_misses
        if self.last_camera_t is not None and t - self.last_camera_t <= cfg.camera_keep_s:
            limit = max(limit, cfg.hard_miss_factor * cfg.max_misses)         # the camera still sees it
        return self.misses >= limit

    # ---- outputs ----------------------------------------------------------------------------------------------------
    @property
    def hit_ratio(self) -> float:
        """Share of the last `existence_window` scans (or of all scans of a younger track) with a LiDAR measurement."""
        return sum(self.history) / len(self.history)

    @property
    def reliability(self) -> float:
        """The hit ratio before the current run of misses: how reliably the track was seen until it was lost (the misses of
        the run itself must not take the bonus away while it is being used)."""
        n = len(self.history) - self.misses
        return sum(self.history) / n if n > 0 else 0.0

    @property
    def reported(self) -> bool:
        """Confirmed and coasting tracks go on to the risk stage."""
        return self.state != TrackState.TENTATIVE

    @property
    def position(self) -> np.ndarray:
        return self.filter.position

    @property
    def velocity(self) -> np.ndarray:
        return self.filter.velocity