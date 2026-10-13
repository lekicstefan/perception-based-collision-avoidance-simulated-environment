"""The tracker (step 6.5): association, lifecycle, split and merge handling, per LiDAR scan.

`Tracker.step(clusters, t)` takes the world clusters of one scan (`clusters_to_world`) and does, in this order:

  1. measurement   every cluster becomes (z, R, t) with the chosen point (default: the corrected centre, step 6.3). With the size
                   memory (step 6.11) the centre is completed to the extent the track remembers, so it is measured per track
  2. association   gate and Hungarian assignment (step 6.4) against the existing tracks
  3. update        assigned tracks are updated with their measurement (a hit); a tentative track with `confirm_hits`
                   consecutive hits is confirmed
  4. misses        tracks without a measurement predict only (coast); too many misses delete them
  5. merge         a track that gets no measurement of its own but is explained by the cluster of another track (the gate
                   would accept that cluster for it) in `merge_scans` scans in a row is taken to have fused with it; the track
                   with fewer hits is deleted ("two objects whose clusters fuse")
  6. births        every unassigned measurement starts a new tentative track ("the other part of a split cluster")

`Tracker.step_camera(rgb, pose, t, camera)` is the second input, at the camera rate and with the camera time stamp (step 6.6):
every reported track gets a bearing measurement from the edges in its predicted image region (bearing.py) and an extended
Kalman update. It does not create, confirm or delete tracks and does not reset the misses of the LiDAR lifecycle; it counts
`track.camera_hits`, the evidence that a coasting track is still in the picture.

Nothing here knows about classes or sizes. The step returns what happened so that tools and tests can follow the tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from server.tracking.association import AssociationConfig, associate
from server.tracking.extent import ExtentConfig, centre_with_extent, grow, hull, initial_extent, with_centre
from server.tracking.bearing import BearingConfig, CameraResult, measure_bearing, prepare_frame
from server.tracking.kalman import ConstantVelocityFilter, FilterConfig
from server.tracking.measurement import WorldCluster
from server.tracking.noise import MeasurementNoise, measurement_of
from server.tracking.outputs import OutputConfig, TrackOutput, track_output, update_motion_state
from server.tracking.track import Track, TrackConfig, TrackState


@dataclass(frozen=True)
class TrackerConfig:
    point: str = "centre"                                   # nearest, centre or centroid (step 6.3)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    bearing: BearingConfig = field(default_factory=BearingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    extent: ExtentConfig = field(default_factory=ExtentConfig)        # size memory of the tracks (step 6.11; only for point 'centre')


@dataclass
class StepResult:
    t: float
    matches: list = field(default_factory=list)             # [(track id, cluster index, squared Mahalanobis distance)]
    births: list = field(default_factory=list)              # [(track id, cluster index)]
    confirmed: list = field(default_factory=list)           # track ids confirmed in this step
    deleted: list = field(default_factory=list)             # track ids deleted in this step (misses)
    merged: list = field(default_factory=list)              # [(deleted track id, track id it was merged into)]


class Tracker:
    def __init__(self, config: TrackerConfig = TrackerConfig(), noise: MeasurementNoise = MeasurementNoise()):
        self.config = config
        self.noise = noise
        self.tracks: list[Track] = []
        self._next_id = 1
        self._dups: dict = {}                                    # (id a, id b) -> scans in which the two tracks were at the same place

    @property
    def reported(self) -> list:
        """Confirmed and coasting tracks."""
        return [tr for tr in self.tracks if tr.reported]

    def step(self, clusters: list, t: float) -> StepResult:
        cfg = self.config
        out = StepResult(t)
        meas = [measurement_of(w, self.noise, cfg.point) for w in clusters]
        ecfg = cfg.extent
        use_extent = ecfg.enabled and cfg.point == "centre"
        pair_cache = {}

        def pair(i, j):                                                  # cluster j as track i sees it (its remembered extent)
            if (i, j) not in pair_cache:
                ext = self.tracks[i].eff_extent
                ext = initial_extent(clusters[j], ecfg) if ext is None else ext
                pair_cache[(i, j)] = measurement_of(with_centre(clusters[j], ext, ecfg), self.noise, cfg.point)
            return pair_cache[(i, j)]

        res = associate([tr.filter for tr in self.tracks], meas, cfg.association, pair if use_extent else None)
        owner = {}                                                       # cluster index -> track index that got it
        for i, j in res.matches:
            tr = self.tracks[i]
            z, r, mt = meas[j]
            if use_extent:
                w = clusters[j]
                old = initial_extent(w, ecfg) if tr.eff_extent is None else tr.eff_extent        # the extent the track's point is defined with now
                lo, hi = hull(w)
                tr.recent_hulls = (tr.recent_hulls + [hi - lo])[-max(ecfg.persist_scans, 1):]
                mem = grow(initial_extent(w, ecfg) if tr.extent is None else tr.extent, w, ecfg, tr.recent_hulls)
                new = np.minimum(np.maximum(mem, hi - lo), ecfg.max_extent_m)                     # memory, but never less than what is visible now
                if not np.array_equal(new, old):                          # a new definition of the point: move the track, not the filter's belief of motion
                    tr.rereference(centre_with_extent(w, new, ecfg) - centre_with_extent(w, old, ecfg))
                tr.extent, tr.eff_extent = mem, new
                z, r, mt = measurement_of(with_centre(w, new, ecfg), self.noise, cfg.point)
            if tr.hit(clusters[j], z, r, mt):
                out.confirmed.append(tr.id)
            out.matches.append((tr.id, j, float(res.d2[i, j])))
            owner[j] = i
        dead = set()
        for i in res.unmatched_tracks:
            tr = self.tracks[i]
            if tr.miss(t):
                dead.add(i)
                out.deleted.append(tr.id)
                continue
            fits = [j for j in owner if res.d2[i, j] <= cfg.association.gate_chi2]      # another track's cluster explains it
            tr.shared = tr.shared + 1 if fits else 0
            if tr.shared >= cfg.track.merge_scans:
                keep = self.tracks[owner[min(fits, key=lambda j: res.d2[i, j])]]
                if tr.hits <= keep.hits:
                    dead.add(i)
                    out.merged.append((tr.id, keep.id))
        self.tracks = [tr for i, tr in enumerate(self.tracks) if i not in dead]
        self._merge_duplicates(out)
        for j in res.unmatched_measurements:
            z, r, mt = meas[j]
            ext = initial_extent(clusters[j], ecfg) if use_extent else None
            if use_extent:
                z, r, mt = measurement_of(with_centre(clusters[j], ext, ecfg), self.noise, cfg.point)
            tr = Track(self._next_id, ConstantVelocityFilter.initialise(z, r, mt, cfg.filter), clusters[j], cfg.track)
            tr.extent = tr.eff_extent = ext
            tr.recent_hulls = [] if ext is None else [ext.copy()]
            self._next_id += 1
            self.tracks.append(tr)
            out.births.append((tr.id, j))
        for tr in self.tracks:
            update_motion_state(tr, cfg.output)
        return out

    def _merge_duplicates(self, out: StepResult) -> None:
        """Two tracks of one object (step 6.11): the same cluster goes alternately to the one and to the other, so the track without
        a measurement never stays unexplained for `merge_scans` scans in a row and the merge test of step 5 never fires. Here the
        tracks are compared with each other: at the same place (closer than `duplicate_max_dist_m`, within `duplicate_chi2` of
        their position covariances, similar velocity) in `duplicate_scans` scans, not necessarily in a row, they are one object
        and the one with fewer hits is deleted. The distance limit keeps two objects that touch (a person at the side of a car)
        apart: their centres are further apart than this."""
        cfg = self.config.track
        if cfg.duplicate_scans <= 0:
            return
        alive = {tr.id for tr in self.tracks}
        self._dups = {k: v for k, v in self._dups.items() if k[0] in alive and k[1] in alive}
        dead = set()
        for a in range(len(self.tracks)):
            for b in range(a + 1, len(self.tracks)):
                ta, tb = self.tracks[a], self.tracks[b]
                key = (min(ta.id, tb.id), max(ta.id, tb.id))
                if ta.id in dead or tb.id in dead:
                    continue
                d = ta.filter.x[:2] - tb.filter.x[:2]
                dist = float(np.hypot(*d))
                same = False
                if dist < cfg.duplicate_max_dist_m and np.hypot(*(ta.filter.x[2:4] - tb.filter.x[2:4])) < cfg.duplicate_max_dv_mps:
                    S = ta.filter.P[:2, :2] + tb.filter.P[:2, :2]
                    same = float(d @ np.linalg.solve(S, d)) < cfg.duplicate_chi2
                n = self._dups.get(key, 0)
                n = n + 1 if same else max(n - 1, 0)
                self._dups[key] = n
                if n >= cfg.duplicate_scans:
                    drop = ta if (ta.hits, -ta.id) < (tb.hits, -tb.id) else tb
                    keep = tb if drop is ta else ta
                    dead.add(drop.id)
                    out.merged.append((drop.id, keep.id))
        if dead:
            self.tracks = [tr for tr in self.tracks if tr.id not in dead]

    def outputs(self) -> list:
        """The derived outputs (outputs.py) of the confirmed and coasting tracks."""
        return [track_output(tr, self.config.output) for tr in self.reported]

    def step_camera(self, rgb, pose, t: float, camera, diagnostics: list | None = None) -> list:
        """One camera frame captured at time t with the car at `pose`. Returns a CameraResult per reported track. When a list is
        given as `diagnostics`, the BearingMeasurement of every track (with its ROI and expected extent) is appended to it."""
        cfg = self.config.bearing
        if not cfg.enabled:
            return []
        frame = prepare_frame(rgb, cfg)
        out = []
        for tr in self.tracks:
            if not tr.reported:
                continue
            m = measure_bearing(tr, frame, camera, pose, t, cfg)
            if diagnostics is not None:
                diagnostics.append(m)
            if m.theta is None:
                out.append(CameraResult(tr.id, m.reason))
                continue
            d2 = tr.filter.gate_distance2_bearing(m.theta, m.sigma, m.cam_xy, t)
            if d2 > cfg.gate_chi2:
                out.append(CameraResult(tr.id, "gated", nis=d2))
                continue
            r = tr.filter.update_bearing(m.theta, m.sigma, m.cam_xy, t)
            if r.applied:
                tr.camera_hit(t)
                out.append(CameraResult(tr.id, r.reason, float(r.innovation[0]), r.nis))
            else:
                out.append(CameraResult(tr.id, r.reason))
        return out