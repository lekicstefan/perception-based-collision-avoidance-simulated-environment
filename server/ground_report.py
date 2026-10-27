import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from server.communication import protocol as P
from server.geometry.calibration import Calibration
from server.perception.building import BUILDING
from server.perception.ground import GROUND, NONE, OTHER, REASON_NAMES
from server.perception.range_image import LidarGeometry

COLOURS = {NONE: "#bbbbbb", GROUND: "#2e9e4f", OTHER: "#d62728", BUILDING: "#1f5fd6"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--times", default=None, help="simulation times in seconds, for example 5,12.5,30: one picture of the kept scan nearest to each")
    ap.add_argument("--from", dest="start", type=float, default=None, help="first time (s) of a series of pictures, with --step")
    ap.add_argument("--to", dest="stop", type=float, default=None, help="last time (s) of the series, default the end of the run")
    ap.add_argument("--step", type=float, default=None, help="seconds between the pictures of the series")
    ap.add_argument("--tint", type=float, default=0.40, help="how strongly the camera picture is tinted by the labels, 0 to 1")
    ap.add_argument("--reach", type=float, default=6.0, help="pixels around a lidar point that take its tint")
    args = ap.parse_args()
    run = Path(args.run) / "python"
    stages, frames, building = run / "stages.csv", run / "ground_frames.npz", run / "building_surfaces.npz"
    if not stages.exists() or not frames.exists():
        sys.exit(f"no {stages} or {frames}: run the processor first")

    with open(stages, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} scans. Time per stage (ms):")
    print("stage            median     p97     max")
    total = np.zeros(len(rows))
    for name in [k for k in rows[0] if k != "t_ms"]:
        ms = np.array([float(r[name]) for r in rows])
        total += ms
        print(f"{name:14s} {np.median(ms):8.2f} {np.percentile(ms, 97):7.2f} {ms.max():7.2f}")
    print(f"{'all stages':14s} {np.median(total):8.2f} {np.percentile(total, 97):7.2f} {total.max():7.2f}\n")

    z = np.load(frames)
    ranges, labels, reasons, el, az, mount, times = z["ranges"], z["labels"], z["reasons"], z["elevations_deg"], z["azimuths_deg"], z["mount"], z["t"]
    n, nrows, ncols = labels.shape
    print(f"{n} kept scans, {times[0]:.1f} s to {times[-1]:.1f} s. Rows are counted from the bottom (1 = lowest beam). Percent of all cells of the row. No hit = no return or nothing hit within the maximum range.")
    print("row  elevation  flat ground at   ground   other  building   no hit      median distance of ground cells in the 40 middle columns")
    mid = slice(ncols // 2 - 20, ncols // 2 + 20)
    for i in range(nrows - 1, -1, -1):
        lab = labels[:, i]
        pct = [100.0 * np.mean(lab == k) for k in (GROUND, OTHER, BUILDING, NONE)]
        e = np.radians(el[i])
        flat = mount[2] / np.tan(-e) if e < 0 else float("inf")
        road = f"{flat:6.1f} m" if flat <= 100 else "     - "
        g = labels[:, i, mid] == GROUND
        dist = np.cos(e) * ranges[:, i, mid].astype(float) / 100.0
        med = f"{np.median(dist[g]):8.2f} m" if g.any() else "       -"
        print(f"{nrows - i:3d}  {el[i]:7.2f}°   {road}      {pct[0]:6.1f}%  {pct[1]:5.1f}%  {pct[2]:7.1f}%  {pct[3]:8.1f}%   {med}")
    print("\n'flat ground at' is where flat ground would meet the beam, measured from the sensor. Distances are from the sensor, not from the car.")

    r_all = ranges.astype(float) / 100.0
    e_rad, a_rad = np.radians(el)[:, None], np.radians(az)[None, :]
    height = mount[2] + r_all * np.sin(e_rad)
    fwd = mount[0] + r_all * np.cos(e_rad) * np.cos(a_rad)
    lat = mount[1] + r_all * np.cos(e_rad) * np.sin(a_rad)
    hit_all = (ranges != 0) & (ranges != 65535)
    sees = np.zeros(nrows, bool)
    sees[nrows - 18:] = True
    print("\nWhy cells in the 18 rows that can see the ground (rows 1 to 18) were called other, all kept scans:")
    inside = (labels == OTHER) & sees[None, :, None]
    for code in range(1, len(REASON_NAMES)):
        count = int(((reasons == code) & inside).sum())
        if count:
            print(f"  {count:5d}  {REASON_NAMES[code]}")
    print("\nThe first 25 of them in the first kept scan. Heights are above the plane of the car's wheels. 'below' is the next lower beam")
    print("in the same column, 'left' and 'right' are the neighbouring columns in the same row (height, G = ground, O = other, B = building, - = none).")
    print("row  col  azimuth  forward  lateral  height   reason   | below            | left             | right")

    def info(k, i, j):
        if i < 0 or i >= nrows or j < 0 or j >= ncols or not hit_all[k, i, j]:
            return "-"
        return f"{height[k, i, j]:+.2f} {'GOB'[labels[k, i, j] - 1]}"

    for i, j in list(zip(*np.nonzero(inside[0])))[:25]:
        below = next((info(0, ii, j) for ii in range(i + 1, nrows) if hit_all[0, ii, j]), "-")
        print(f"{nrows - i:3d} {j:4d} {az[j]:7.1f}° {fwd[0, i, j]:7.1f}  {lat[0, i, j]:7.1f}  {height[0, i, j]:+6.2f}   {REASON_NAMES[reasons[0, i, j]][:28]:28s} | {below:16s} | {info(0, i, j - 1):16s} | {info(0, i, j + 1)}")

    chosen = choose(times, args)
    if not chosen:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return
    calibration = Calibration.from_json(str(z["calibration"]))
    dirs = LidarGeometry(calibration.lidar).dirs_vehicle
    origin = calibration.lidar.mount.translation()
    cam_w, cam_h, cam_fmt = (int(v) for v in z["camera_format"])
    offsets, cam_data = z["camera_offsets"], z["camera_data"]
    cmap = ListedColormap([COLOURS[k] for k in (NONE, GROUND, OTHER, BUILDING)])
    surface_times = None
    if building.exists():
        b = np.load(building)
        surface_times, link_array, surface_array = b["t"], b["links"], b["surfaces"]
        link_offsets, surface_offsets = np.concatenate([[0], np.cumsum(b["link_counts"])]), np.concatenate([[0], np.cumsum(b["surface_counts"])])
    print(f"\n{len(chosen)} pictures:")
    for k in chosen:
        hit = hit_all[k]
        fig = plt.figure(figsize=(11, 17))
        grid = fig.add_gridspec(4, 2, height_ratios=[0.9, 0.7, 1.5, 0.8])
        ax = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :]), fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[3, :])]
        data = cam_data[offsets[k]:offsets[k + 1]]
        if len(data):
            rgb = P.camera_to_rgb(P.CameraPayload(cam_w, cam_h, P.CameraFormat(cam_fmt), memoryview(data)))
            ax[0].imshow(tint_camera(rgb, labels[k], ranges[k], calibration, dirs, origin, args.tint, args.reach))
            ax[0].set_title(f"camera at t = {times[k]:.2f} s, tinted {args.tint:.0%}: green ground, grey no return or no hit, red other, blue building")
        else:
            ax[0].set_title("no camera frame kept for this scan")
        ax[0].axis("off")
        ax[1].imshow(labels[k], cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        ax[1].set_title("lidar cells (row 0 on top is the highest beam, left is the left of the car)  grey no return or no hit, green ground, red other, blue building")
        for panel, (u, v, name) in zip(ax[2:], [(lat[k], fwd[k], "from above: y (m, positive = left, on the left), x forward (m)"),
                                                (lat[k], fwd[k], "from above, with the surfaces: black = building, orange = removed by the ground check, grey = not a building, numbers are areas in m2"),
                                                (fwd[k], height[k], "from the side: x forward (m), height (m)")]):
            for kk in (GROUND, OTHER, BUILDING):
                m = hit & (labels[k] == kk)
                panel.scatter(u[m], v[m], s=3, color=COLOURS[kk])
            panel.set_title(name)
            panel.grid(alpha=0.3)
        for panel in ax[2:4]:
            panel.set_xlim(25, -25)
            panel.set_ylim(0, 60)
            panel.set_aspect("equal")
        if surface_times is not None:
            s = int(np.argmin(np.abs(surface_times - times[k])))
            links = link_array[link_offsets[s]:link_offsets[s + 1]]
            surfaces = surface_array[surface_offsets[s]:surface_offsets[s + 1]]
            shade = {0: "#888888", 1: "black", 2: "tab:orange"}
            for x0, y0, x1, y1, status in sorted(links.tolist(), key=lambda l: l[4] == 1):
                ax[3].plot([y0, y1], [x0, x1], color=shade[int(status)], linewidth=1.0 if status == 0 else 2.0)
            for cx, cy, area, status in surfaces:
                if area >= 5.0:
                    ax[3].text(cy, cx, f"{area:.0f}", fontsize=7, color=shade[int(status)])
            biggest = sorted(surfaces.tolist(), key=lambda s_: -s_[2])[:8]
            print(f"  t = {times[k]:.2f} s, largest surfaces (m2, * = building, x = removed by the ground check): " + ", ".join(f"{a:.0f}{' *x'[int(st)]}".rstrip() for _, _, a, st in biggest))
        ax[4].set_xlim(0, 60)
        ax[4].set_ylim(-1, 4)
        fig.tight_layout()
        out = run / f"ground_{times[k]:08.2f}s.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"  {out}")


def choose(times: np.ndarray, args) -> list:
    """Indices of the kept scans to draw: the ones nearest to the asked times."""
    asked = []
    if args.times:
        asked += [float(t) for t in args.times.split(",")]
    if args.step is not None or args.start is not None or args.stop is not None:
        if args.step is None or args.step <= 0:
            sys.exit("--from and --to need a --step above 0")
        start = times[0] if args.start is None else args.start
        stop = times[-1] if args.stop is None else args.stop
        asked += list(np.arange(start, stop + 1e-9, args.step))
    if not asked:
        n = len(times)
        return sorted({0, n // 2, n - 1})
    return sorted({int(np.argmin(np.abs(times - t))) for t in asked})


def tint_camera(rgb, labels, ranges, calibration, dirs, origin, tint, reach):
    """Each lidar cell is projected into the camera picture (a cell with no return at the maximum range along its ray).
    Every pixel within reach pixels of a projected cell takes the tint of the nearest one."""
    from scipy.spatial import cKDTree
    h, w = rgb.shape[:2]
    hit = (ranges != 0) & (ranges != 65535)
    r = np.where(hit, ranges.astype(float) / 100.0, calibration.lidar.max_range_m)
    points = (origin + r[..., None] * dirs).reshape(-1, 3)
    u, v, depth = calibration.camera.project(points)
    ok = (depth > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    lab = labels.reshape(-1)[ok]
    tree = cKDTree(np.column_stack([u[ok], v[ok]]))
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dist, idx = tree.query(np.column_stack([xs.ravel(), ys.ravel()]), distance_upper_bound=reach)
    near = np.isfinite(dist)
    colour = np.zeros((h * w, 3))
    for kk, hexcolour in COLOURS.items():
        colour[near & (lab[np.minimum(idx, len(lab) - 1)] == kk)] = [int(hexcolour[i:i + 2], 16) for i in (1, 3, 5)]
    out = rgb.reshape(-1, 3).astype(float)
    out[near] = (1 - tint) * out[near] + tint * colour[near]
    return out.reshape(h, w, 3).astype(np.uint8)


if __name__ == "__main__":
    main()