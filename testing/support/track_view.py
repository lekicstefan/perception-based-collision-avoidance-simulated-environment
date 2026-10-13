"""The tracker on top of the top view of a scan (step 6.11, used by testing/tools/perception/render_scan.py --tracks).

`overlay(tracker, step_result, world_clusters, pose)` turns the state of the tracker after one scan into what
`scan_view.top_view` draws: the tracks (position, velocity, the 95 % ellipse of the position, the extent the track remembers as a
box), a line from every cluster to the track that got it, and a headline with the numbers and the events of the scan (births, merges,
deletions). The tracker works in the world frame and the top view is the vehicle frame of the scan, so everything is moved with the pose
of the scan.
"""
from __future__ import annotations

import numpy as np

from server.geometry.frames import rotate_to_vehicle, world_to_vehicle
from server.tracking.outputs import track_output


def overlay(tracker, res, world: list, pose, n_ellipse: int = 40):
    """(tracks, links, headline) for scan_view.top_view; positions in the vehicle frame."""
    cfg = tracker.config.output
    phi = np.linspace(0.0, 2.0 * np.pi, n_ellipse, endpoint=False)
    tracks, by_id = [], {}
    for tr in tracker.tracks:
        o = track_output(tr, cfg)
        major, minor, ang = o.ellipse
        c, s = np.cos(ang), np.sin(ang)
        ell = np.asarray(o.position) + np.stack([major * np.cos(phi) * c - minor * np.sin(phi) * s,
                                                  major * np.cos(phi) * s + minor * np.sin(phi) * c], axis=1)
        box = None
        ext = getattr(tr, "eff_extent", None)
        if ext is not None:
            h = np.asarray(ext, dtype=float) / 2.0
            box = world_to_vehicle(np.asarray(o.position) + np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]]), pose)
        label = f"T{tr.id} {o.speed:.1f}m/s" + (" S" if o.static else "")
        pos = world_to_vehicle(np.asarray(o.position), pose)
        by_id[tr.id] = pos
        tracks.append(dict(id=tr.id, state=o.state, static=bool(o.static), pos=pos, vel=rotate_to_vehicle(np.asarray(o.velocity), pose),
                           ellipse=world_to_vehicle(ell, pose), box=box, label=label))
    links = [(tuple(world_to_vehicle(np.asarray(world[j].centre_xy), pose)), tuple(by_id[tid]))
             for tid, j, _ in res.matches if tid in by_id]
    n = {k: sum(1 for t in tracks if t["state"] == k) for k in ("confirmed", "tentative", "coasting")}
    events = []
    if res.births:
        events.append("born " + ",".join(f"T{t}" for t, _ in res.births))
    if res.merged:
        events.append("merged " + ",".join(f"T{a}>T{b}" for a, b in res.merged))
    if res.deleted:
        events.append("deleted " + ",".join(f"T{t}" for t in res.deleted))
    head = f"t={res.t:.2f}s tracks: {n['confirmed']} confirmed {n['tentative']} tentative {n['coasting']} coasting" + ("   " + "; ".join(events) if events else "")
    return tracks, links, head