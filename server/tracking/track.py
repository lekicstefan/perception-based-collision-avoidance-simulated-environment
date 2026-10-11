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
The track also remembers the largest extent it has been seen with (a partial view of a big object is then recognisable).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from server.tracking.kalman import ConstantVelocityFilter
from server.tracking.measurement import WorldCluster


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
        self.camera_hits = 0            # camera bearing updates applied (step 6.6); evidence of existence for the risk stage
        self.last_camera_t = None
        self.shared = 0                 # consecutive scans in which another track's cluster also fits this track (merge test)
        self.last_hit_t = flt.t
        self.last_cluster = w
        self.max_width_m = w.width_m
        self.max_depth_m = w.depth_m
        self.max_height_m = w.height_m

    # ---- events -----------------------------------------------------------------------------------------------------
    def hit(self, w: WorldCluster, z, r, t: float) -> bool:
        """A measurement was assigned. Returns True when this hit confirmed the track."""
        self.filter.update(z, r, t)
        self.hits += 1
        self.consecutive_hits += 1
        self.misses = 0
        self.shared = 0
        self.age_scans += 1
        self.last_hit_t = t
        self.last_cluster = w
        self.max_width_m = max(self.max_width_m, w.width_m)
        self.max_depth_m = max(self.max_depth_m, w.depth_m)
        self.max_height_m = max(self.max_height_m, w.height_m)
        was = self.state
        if self.state == TrackState.COASTING:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.TENTATIVE and self.consecutive_hits >= self.config.confirm_hits:
            self.state = TrackState.CONFIRMED
        return was == TrackState.TENTATIVE and self.state == TrackState.CONFIRMED

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
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.COASTING
        if self.state == TrackState.TENTATIVE:
            return self.misses > self.config.tentative_max_misses
        return self.misses >= self.config.max_misses

    # ---- outputs ----------------------------------------------------------------------------------------------------
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