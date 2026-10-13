"""Sensitivity analysis of the speed caps to the risk parameters (step 7.9): is the risk layer measurable before the experiments?

    python -m testing.tools.risk.sensitivity_check [--out testing/data/sensitivity_check]

Writes summary.txt, CSV files and three pictures to the output folder:
  sweeps.png         both caps against every parameter, one at a time, the others at the medium bundle (hazard in the path, and a lead vehicle)
  levels.png         both caps against the distance for the three risk levels
  tornado.png        the change of each cap when only one parameter goes from its low-risk to its high-risk value
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from server.risk.risk_level import BUNDLES, RiskLevel
from testing.support import risk_sensitivity as rs
from testing.support.paths import DATA_DIR

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
COMFORT, SAFETY = "#2a78d6", "#eb6834"                      # categorical slots 1 and 2 of the reference palette


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plot_sweeps(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(rs.PARAMETERS)
    fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 6.0), facecolor=SURFACE, sharey="row")
    for r, scene in enumerate((rs.STATIC, rs.LEAD)):
        for c, name in enumerate(names):
            ax = axes[r, c]
            _style(ax)
            data = np.array(rs.sweep(name, scene))
            ax.plot(data[:, 0], data[:, 1], color=COMFORT, linewidth=2, label="comfort cap")
            ax.plot(data[:, 0], data[:, 2], color=SAFETY, linewidth=2, label="safety cap")
            if name == "epsilon":
                ax.set_xscale("log")
            for lvl in RiskLevel:                                 # where the three levels sit on the axis
                v = getattr(BUNDLES[lvl].apply(), rs.PARAMETERS[name][0], None)
                if v is not None and name != "delay":
                    ax.axvline(v, color=MUTED, linewidth=0.6, linestyle=":")
            ax.set_xlabel(rs.PARAMETERS[name][2], fontsize=7, color=INK)
            if c == 0:
                ax.set_ylabel(f"{scene.name}\ncap (m/s)", fontsize=8, color=INK)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Speed caps against one risk parameter at a time (others at the medium bundle; dotted lines: the levels)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_levels(path: Path, distances):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE, sharey=True)
    for ax, scene in zip(axes, (rs.STATIC, rs.LEAD)):
        _style(ax)
        curves = rs.level_curves(distances, scene)
        for level, ls in (("low", "-"), ("medium", "--"), ("high", ":")):
            d = np.array(curves[level])
            ax.plot(d[:, 0], d[:, 2], color=SAFETY, linewidth=2, linestyle=ls)
            ax.plot(d[:, 0], d[:, 1], color=COMFORT, linewidth=2, linestyle=ls)
            ax.annotate(level, (d[-1, 0], d[-1, 2]), xytext=(3, 0), textcoords="offset points", fontsize=7, color=MUTED, va="center")
        ax.set_title(scene.name, fontsize=9, color=INK)
        ax.set_xlabel("distance to the hazard d_path (m)", fontsize=8, color=INK)
    axes[0].set_ylabel("speed cap (m/s)", fontsize=8, color=INK)
    axes[0].plot([], [], color=COMFORT, linewidth=2, label="comfort cap")
    axes[0].plot([], [], color=SAFETY, linewidth=2, label="safety cap")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_tornado(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = rs.tornado(rs.LEAD)                                    # the lead scene shows every parameter, including a_h
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.18, [r[3] for r in rows], height=0.34, color=COMFORT, label="comfort cap")
    ax.barh(y - 0.18, [r[4] for r in rows], height=0.34, color=SAFETY, label="safety cap")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("change of the cap (m/s) when the parameter alone goes from its low-risk to its high-risk value", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def summary_lines() -> list:
    out = ["Sensitivity of the speed caps to the risk parameters (step 7.9)", ""]
    for scene in (rs.STATIC, rs.LEAD):
        e = rs.level_effect(scene)
        out += [f"{scene.name}: d_path {scene.d_path_m:g} m, sigma_d {scene.sigma_d_m:g} m" + (f", v_h {scene.v_hazard_mps:g} m/s, sigma_v {scene.sigma_v_mps:g}" if scene.v_hazard_mps else ""),
                *(f"  {l:7s} comfort cap {e[l][0]:5.2f}  safety cap {e[l][1]:5.2f} m/s" for l in ("low", "medium", "high")),
                f"  low -> high: comfort cap -{100 * e['drop_comfort']:.0f} %, safety cap -{100 * e['drop_safety']:.0f} %", ""]
    out += ["One parameter alone, low-risk -> high-risk value (lead scene, change of the cap in m/s):",
            f"  {'parameter':14s} {'low':>7s} {'high':>7s} {'comfort':>9s} {'safety':>9s}"]
    out += [f"  {n:14s} {a:7.3g} {b:7.3g} {dc:+9.2f} {ds:+9.2f}" for n, a, b, dc, ds in rs.tornado(rs.LEAD)]
    out += ["", "Hazard in the path, the same:"]
    out += [f"  {n:14s} {a:7.3g} {b:7.3g} {dc:+9.2f} {ds:+9.2f}" for n, a, b, dc, ds in rs.tornado(rs.STATIC)]
    out += ["", "Covariance counted in the reachable set (7.3) and in d_eff (7.6): both move the conflict point nearer by kappa z sigma (hazard in the path, 30 m):",
            f"  {'level':8s} {'sigma':>6s} {'extra m':>8s} {'cap once':>9s} {'cap twice':>10s} {'loss':>6s}"]
    out += [f"  {l:8s} {s:6.2f} {x:8.2f} {o:9.2f} {t:10.2f} {o - t:6.2f}" for l, s, x, o, t in rs.double_count()]
    out += ["", "Virtual side hazard (parked car), safety cap with the floor of the level, m/s:",
            f"  {'distance m':>10s} " + " ".join(f"{l.name.lower():>7s}" for l in RiskLevel)]
    out += [f"  {d:10.1f} " + " ".join(f"{rs.virtual_cap(l, d):7.2f}" for l in RiskLevel) for d in (0.5, 2, 5, 10, 20, 40)]
    out += ["", "Lowest ego speed (m/s) at which a hidden pedestrian is NO conflict (station at 20 m, time margin 0.5 s); inf = never:",
            f"  {'gap m':>6s} " + " ".join(f"{v:>7.1f}" for v in (1.5, 2.0, 2.5))]
    out += [f"  {g:6.1f} " + " ".join(f"{rs.no_conflict_speed(g, v, 20.0):7.1f}" for v in (1.5, 2.0, 2.5)) for g in (0.5, 1.0, 1.5, 2.5)]
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DATA_DIR / "sensitivity_check")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for scene in (rs.STATIC, rs.LEAD):
        for name in rs.PARAMETERS:
            write_csv(a.out / f"sweep_{name}_{'lead' if scene.v_hazard_mps else 'static'}.csv", ["value", "comfort_cap", "safety_cap"], rs.sweep(name, scene))
    distances = np.arange(2.0, 80.1, 2.0)
    for scene in (rs.STATIC, rs.LEAD):
        rows = [(lvl, *r) for lvl, c in rs.level_curves(distances, scene).items() for r in c]
        write_csv(a.out / f"levels_{'lead' if scene.v_hazard_mps else 'static'}.csv", ["level", "d_path", "comfort_cap", "safety_cap"], rows)
    write_csv(a.out / "tornado.csv", ["parameter", "low_value", "high_value", "d_comfort", "d_safety"], rs.tornado(rs.LEAD))
    write_csv(a.out / "double_count.csv", ["level", "sigma", "extra_m", "cap_once", "cap_twice"], rs.double_count())
    plot_sweeps(a.out / "sweeps.png")
    plot_levels(a.out / "levels.png", distances)
    plot_tornado(a.out / "tornado.png")
    text = "\n".join(summary_lines()) + "\n"
    (a.out / "summary.txt").write_text(text)
    print(text)
    print("wrote", a.out / "summary.txt")
    return a.out


if __name__ == "__main__":
    main()