"""Metrics of the perception against the ground truth (step 5.7): recall, precision, position error, fragmentation.

Association. A cluster belongs to a hazard when at least min_inside of its points lie inside the hazard's footprint,
enlarged by margin_m (LiDAR noise, the rim removed with the ground). A cluster that fits several hazards goes to the one
holding most of its points.

Which hazards count for recall. A hazard has to be detectable: in view, and at least min_hits of its cells returned
(visibility.csv). A hazard hit by 1 cell is not a target a segmentation can be expected to find, and the minimum cluster
size would remove it anyway. Recall is also reported against the number of returned cells, which is what decides it.

Errors (per detected hazard, from the cluster that holds most of its points):
  centre_err     distance from the corrected box centre to the true box centre (ground plane)
  centroid_err   the same for the plain centroid (shown for comparison only)
  nearest_err    range of the cluster's nearest point minus the true ground-plane distance sensor -> box. Always >= 0
                 up to noise, since the box is only sampled by cells. This is the conservative collision quantity.
  width_err      visible width of the cluster minus the true width seen across the line of sight
Precision = matched clusters / all clusters. Clusters that match no hazard are objects the ground truth does not list
(terrain, an obstacle that is not a hazard) or false alarms; the number is reported, not judged.
Fragmentation = detected hazards that are made of more than one cluster.
Merged = a hazard with no cluster of its own whose points sit in a cluster that belongs to a neighbour (two objects touching
at similar range). It counts as missed in the recall, and is also reported on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RANGE_BINS = [0.0, 20.0, 40.0, 60.0, 80.0, np.inf]
HIT_BINS = [0, 1, 2, 3, 6, 11, np.inf]            # returned cells: 0, 1, 2, 3-5, 6-10, 11+


@dataclass(frozen=True)
class EvalConfig:
    margin_m: float = 0.4
    min_inside: float = 0.5
    min_hits: int = 2
    diag_margin_m: float = 0.05                  # tight footprint for the 'which clusters hold the hazard's own points' column


def associate(clusters: list, truths: list, cfg: EvalConfig) -> list:
    """For every cluster the index of its hazard (or None)."""
    out = []
    for c in clusters:
        xy = c.points[:, :2]
        frac = [float(t.contains(xy, cfg.margin_m).mean()) for t in truths]
        k = -1
        if frac:
            best = max(frac)
            tied = [i for i, f in enumerate(frac) if f >= best - 1e-9]          # two touching hazards both hold it: the nearer centre
            k = min(tied, key=lambda i: float(np.linalg.norm(xy.mean(axis=0) - truths[i].centre)))
        out.append(k if k >= 0 and frac[k] >= cfg.min_inside else None)
    return out


def _edge_samples(t, n: int = 12) -> np.ndarray:
    c = t.footprint()
    return np.array([c[i] + (c[(i + 1) % 4] - c[i]) * f for i in range(4) for f in np.linspace(0.0, 1.0, n)])


def _nearest_other(k: int, truths: list, sensor_xy):
    """(id, gap between the footprints in m, difference of the distances from the sensor) of the closest other hazard."""
    if len(truths) < 2:
        return -1, float("nan"), float("nan")
    mine = _edge_samples(truths[k])
    best = (-1, float("inf"), float("nan"))
    for j, o in enumerate(truths):
        if j == k:
            continue
        gap = float(np.min(np.linalg.norm(mine[:, None, :] - _edge_samples(o)[None, :, :], axis=2)))
        if gap < best[1]:
            best = (o.id, gap, abs(truths[k].distance_from(sensor_xy) - o.distance_from(sensor_xy)))
    return best


def evaluate_scan(scan: int, t: float, clusters: list, truths: list, sensor_xy, cfg: EvalConfig = EvalConfig()):
    """(rows per hazard, rows per cluster) of one scan, as lists of dicts.

    Besides the metrics every hazard row says what is in its footprint, to tell the causes of a miss apart:
      inside_pts   cluster points inside the footprint, from any cluster
      stolen_pts   those that belong to a cluster owned by another hazard (the hazard is merged into a neighbour)
      in_clusters  'cells:points' of every cluster holding points of the hazard itself (tight footprint, 5 cm): a big
                   cluster means it is merged into something, none means it never became a cluster
      owners       ids of the hazards that own clusters reaching into this footprint
      cluster_sizes  cells of the clusters this hazard owns
      other_id, gap_other_m, step_other_m   the closest other hazard, the gap between the two footprints and the
                   difference of their distances from the sensor (a small step is what the split test cannot see)
    A hazard with inside_pts 0 and no cluster was removed before the clusters existed (ground removal, size filter)."""
    owner = associate(clusters, truths, cfg)
    haz_rows = []
    for k, tr in enumerate(truths):
        mine = [i for i, o in enumerate(owner) if o == k]
        true_near = tr.distance_from(sensor_xy)
        inside = [int(tr.contains(c.points[:, :2], cfg.margin_m).sum()) for c in clusters]
        stolen = sum(n for i, n in enumerate(inside) if owner[i] is not None and owner[i] != k)
        owners = sorted({truths[owner[i]].id for i, n in enumerate(inside) if n and owner[i] is not None and owner[i] != k})
        other_id, gap, step = _nearest_other(k, truths, sensor_xy)
        tight = [int(tr.contains(c.points[:, :2], cfg.diag_margin_m).sum()) for c in clusters]
        in_clusters = "|".join(f"{int(c.n_cells)}:{n}" for c, n in zip(clusters, tight) if n)     # cells:own points
        row = dict(scan=scan, t=t, id=tr.id, label=tr.label, hits=tr.hits_returned, in_view=tr.in_view,
                   detectable=bool(tr.in_view and tr.hits_returned is not None and tr.hits_returned >= cfg.min_hits),
                   true_range_m=true_near, true_width_m=tr.lateral_extent(sensor_xy), n_clusters=len(mine), detected=bool(mine),
                   inside_pts=int(sum(inside)), stolen_pts=int(stolen), owners="|".join(map(str, owners)),
                   cluster_sizes="|".join(str(int(clusters[i].n_cells)) for i in mine), other_id=other_id,
                   gap_other_m=gap, step_other_m=step, in_clusters=in_clusters, own_pts=int(sum(tight)))
        row["merged"] = bool(not mine and stolen >= 2)
        if mine:
            best = max(mine, key=lambda i: int(tr.contains(clusters[i].points[:, :2], cfg.margin_m).sum()))
            c = clusters[best]
            near = float(np.linalg.norm(np.asarray(c.nearest_point[:2]) - np.asarray(sensor_xy)))
            row.update(centre_err=float(np.linalg.norm(c.centre_xy - tr.centre)),
                       centroid_err=float(np.linalg.norm(c.centroid_xy - tr.centre)),
                       nearest_err=near - true_near, width_err=float(c.width_m) - row["true_width_m"],
                       n_cells=int(c.n_cells))
        haz_rows.append(row)
    cl_rows = [dict(scan=scan, t=t, label=int(c.label), n_cells=int(c.n_cells), matched=owner[i] is not None,
                    hazard_id=(truths[owner[i]].id if owner[i] is not None else -1),
                    range_m=float(np.linalg.norm(np.asarray(c.nearest_point[:2]) - np.asarray(sensor_xy))))
               for i, c in enumerate(clusters)]
    return haz_rows, cl_rows


def _binned(df: pd.DataFrame, column: str, edges, labels=None) -> pd.DataFrame:
    cut = pd.cut(df[column], bins=edges, right=False, labels=labels)
    return df.groupby(cut, observed=False)


def summarise(haz: pd.DataFrame, clusters: pd.DataFrame) -> dict:
    """Numbers of a whole run. haz and clusters are DataFrames of the rows of evaluate_scan."""
    det = haz[haz["detectable"]]
    found = det[det["detected"]]
    out = {"scans": int(haz["scan"].nunique()) if len(haz) else 0,
           "detectable_hazard_scans": int(len(det)),
           "recall": float(det["detected"].mean()) if len(det) else float("nan"),
           "precision": float(clusters["matched"].mean()) if len(clusters) else float("nan"),
           "clusters": int(len(clusters)), "unmatched_clusters": int((~clusters["matched"]).sum()) if len(clusters) else 0,
           "fragmented_fraction": float((found["n_clusters"] > 1).mean()) if len(found) else float("nan"),
           "merged_fraction": float(det["merged"].mean()) if len(det) and "merged" in det else float("nan")}
    for col in ("centre_err", "centroid_err", "nearest_err", "width_err"):
        if col in found and len(found):
            v = found[col].to_numpy(dtype=float)
            out[col] = {"median": float(np.median(v)), "p90": float(np.percentile(np.abs(v), 90)), "max_abs": float(np.abs(v).max())}
    if len(det):
        out["recall_by_range"] = {str(k): (float(g["detected"].mean()), int(len(g)))
                                  for k, g in _binned(det, "true_range_m", RANGE_BINS) if len(g)}
        out["recall_by_hits"] = {str(k): (float(g["detected"].mean()), int(len(g)))
                                 for k, g in _binned(haz[haz["in_view"] & haz["hits"].notna()], "hits", HIT_BINS) if len(g)}
        out["recall_by_label"] = {str(k): (float(g["detected"].mean()), int(len(g))) for k, g in det.groupby("label")}
    if len(found):
        out["fragmented_by_label"] = {str(k): (float((g["n_clusters"] > 1).mean()), int(len(g))) for k, g in found.groupby("label")}
    return out


def format_summary(name: str, s: dict) -> str:
    lines = [f"== {name}",
             f"scans {s['scans']}, detectable hazard-scans {s['detectable_hazard_scans']}, recall {s['recall']:.3f}, "
             f"precision {s['precision']:.3f} ({s['unmatched_clusters']} of {s['clusters']} clusters match no hazard), "
             f"fragmented {s['fragmented_fraction']:.3f}, merged into a neighbour {s['merged_fraction']:.3f} (counted as missed)"]
    for col, what in (("centre_err", "corrected centre error (m)"), ("centroid_err", "centroid error (m)"),
                      ("nearest_err", "nearest-point range error (m)"), ("width_err", "width error (m)")):
        if col in s:
            v = s[col]
            lines.append(f"  {what:34s} median {v['median']:+.3f}   90th pct of |e| {v['p90']:.3f}   max |e| {v['max_abs']:.3f}")
    for key, title in (("recall_by_range", "recall by true range (m)"), ("recall_by_hits", "recall by returned cells"),
                       ("recall_by_label", "recall by hazard"), ("fragmented_by_label", "fragmented by hazard")):
        if key in s:
            lines.append(f"  {title}: " + ", ".join(f"{k} {r:.2f} (n={n})" for k, (r, n) in s[key].items()))
    return "\n".join(lines)