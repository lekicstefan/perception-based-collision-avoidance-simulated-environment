"""Size memory of a track (step 6.11): the tracked point must not move when more or less of the object comes into view.

The corrected centre of perception (cluster_geometry.py) is built in the frame of the line of sight (u from the sensor to the
cluster, v = u turned 90 degrees to the left):

    centre = sensor + u * (depth of the nearest part + D / 2) + v * (middle of the visible extent across)      D = max(visible depth, size prior)

D stands for the depth of the object along the line of sight, which the sensor cannot see. With a size prior of 1 m an object of
4.5 m seen from the front has its centre 1.6 m too near, and the centre jumps when the sensor sees more of the side (the filter
reads the jump as velocity). Here D comes from a memory in the track: the largest extent the object has been seen with.

    hull        axis-aligned box (world axes) around the visible part of a cluster (the corners of its vehicle-frame box and its
                nearest point)
    extent      remembered (Lx, Ly) = largest hull sizes seen so far; grows by at most `max_growth_m` per update and axis (a
                cluster that fused with a neighbour cannot add metres at once) and never beyond `max_extent_m`
    D           max(prior, |ux| Lx + |uy| Ly): the depth along the line of sight of a box of the remembered size (the box
                is taken axis-aligned in the world frame, which is exact for an object parallel to the axes)

The tracker passes the extent in use, which is never smaller than the hull visible now (Track.eff_extent: the memory is held back for a few
scans by the persistence rule below, and a centre inside the visible part would jump from scan to scan). Only D changes: the position of the nearest part and the middle across the line of sight are what perception measured, so for a
track that has seen nothing but small things the centre is the one of perception (a person, a crate: unchanged). A larger D
only moves the centre away from the sensor, along the line of sight. Nothing here knows what the object is: the memory is the
largest size seen, never a class or a category.

When the memory grows the tracked point itself moves (a change of its definition, not a motion of the object): the track is
shifted by the change of the centre (`Track.rereference`) before the measurement is applied, so that the filter does not see it
as velocity. Gating uses the centre under the track's old memory (the point the track currently stands for).
One scan with the cluster of a neighbour (a track swap, a merge) must not stay in the memory for good: the memory grows only to the
size that the last `persist_scans` clusters all showed (found on the dev recording: a person track that took the cluster of the parked
car for one scan kept 1.5 m too much for the rest of its life).
Limits: an object turned against the world axes gets a somewhat too large memory; a cluster that contains two objects makes the
memory too large (the growth cap limits how fast; a reset after a merge is not done).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from server.tracking.measurement import WorldCluster


@dataclass(frozen=True)
class ExtentConfig:
    enabled: bool = True
    prior_m: float = 1.0            # minimum depth D: the size prior of perception (GeometryExtractor.size_prior_m; must be equal to it)
    max_growth_m: float = 1.5       # the remembered extent grows by at most this per update and axis
    max_extent_m: float = 8.0       # and never beyond this
    persist_scans: int = 3          # a size counts only when the last this many clusters were all at least that large (1 = at once)


def hull(w: WorldCluster) -> tuple:
    """(lo, hi): the axis-aligned box (world axes) of the visible part of the cluster, each of shape (2,)."""
    pts = np.vstack([np.asarray(w.corners_xy, dtype=float).reshape(-1, 2), np.asarray(w.nearest_xy, dtype=float)[None]])
    return pts.min(axis=0), pts.max(axis=0)


def initial_extent(w: WorldCluster, cfg: ExtentConfig = ExtentConfig()) -> np.ndarray:
    """The extent of a new track: the hull of its first cluster."""
    lo, hi = hull(w)
    return np.minimum(hi - lo, cfg.max_extent_m)


def grow(extent, w: WorldCluster, cfg: ExtentConfig = ExtentConfig(), recent=None) -> np.ndarray:
    """The remembered extent after seeing cluster w.

    recent: list of the hull sizes (2,) of the clusters of this track, newest last, w included (the tracker keeps it). The
    memory only grows to what the last `persist_scans` clusters all showed, so that one scan in which the track was
    given another object's cluster (a swap, a merge) cannot make the memory too large for good. None: no persistence."""
    extent = np.asarray(extent, dtype=float)
    lo, hi = hull(w)
    size = hi - lo
    if recent is not None:
        if len(recent) < cfg.persist_scans:
            return extent.copy()
        size = np.min(np.asarray(recent[-cfg.persist_scans:], dtype=float), axis=0)
    target = np.minimum(np.maximum(extent, size), cfg.max_extent_m)
    return np.maximum(extent, np.minimum(target, extent + cfg.max_growth_m))


def depth_along(extent, bearing: float, prior_m: float = 1.0) -> float:
    """D: depth along the line of sight (world bearing, rad) of a box of the remembered size, at least the prior."""
    e = np.asarray(extent, dtype=float)
    return float(max(prior_m, abs(np.cos(bearing)) * e[0] + abs(np.sin(bearing)) * e[1]))


def centre_with_extent(w: WorldCluster, extent, cfg: ExtentConfig = ExtentConfig()) -> np.ndarray:
    """The centre of perception with its depth D replaced by the one of the remembered extent."""
    sensor = np.asarray(w.sensor_xy, dtype=float)
    u = np.array([np.cos(w.bearing), np.sin(w.bearing)])
    v = np.array([-u[1], u[0]])
    rel = np.asarray(w.centre_xy, dtype=float) - sensor
    near = float(rel @ u) - max(float(w.depth_m), cfg.prior_m) / 2.0           # depth of the nearest part along u
    return sensor + u * (near + depth_along(extent, w.bearing, cfg.prior_m) / 2.0) + v * float(rel @ v)


def with_centre(w: WorldCluster, extent, cfg: ExtentConfig = ExtentConfig()) -> WorldCluster:
    """The cluster with its tracked centre replaced by the centre of the remembered extent."""
    return replace(w, centre_xy=centre_with_extent(w, extent, cfg))