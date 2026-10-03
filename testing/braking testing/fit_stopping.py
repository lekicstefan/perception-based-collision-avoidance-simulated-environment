import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A_MAX = 8.0  # real maximum braking of the simulated car (m/s^2)

here = Path(__file__).parent
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "stopping_table.csv"
df = pd.read_csv(csv_path)
v = df["v0_ms"].to_numpy()
d = df["distance_m"].to_numpy()

# Fit A: a_max fixed, one parameter t_eff in d = v t + v^2 / (2 a)
t_a = float(np.sum(v * (d - v**2 / (2 * A_MAX))) / np.sum(v**2))
res_a = d - (v * t_a + v**2 / (2 * A_MAX))

# Fit B (check): free intercept, delay and deceleration in d = d0 + t v + c v^2
X = np.vstack([np.ones_like(v), v, v**2]).T
d0, t_b, c = np.linalg.lstsq(X, d, rcond=None)[0]
a_b = 1.0 / (2.0 * c)

out = {
    "source": csv_path.name,
    "n_points": int(len(v)),
    "a_max_assumed": A_MAX,
    "t_eff_s": round(t_a, 4),
    "rmse_m": round(float(np.sqrt(np.mean(res_a**2))), 4),
    "free_fit": {"intercept_m": round(float(d0), 4), "t_s": round(float(t_b), 4), "a_ms2": round(float(a_b), 3)},
}
(here / "stopping_fit.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

vv = np.linspace(0, v.max() * 1.02, 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(v * 3.6, d, "o", label="simulation")
ax[0].plot(vv * 3.6, vv * t_a + vv**2 / (2 * A_MAX), "-", label=f"fit: t = {t_a:.3f} s, a = {A_MAX:g} m/s²")
ax[0].set_xlabel("speed (km/h)")
ax[0].set_ylabel("stopping distance (m)")
ax[0].legend()
ax[1].plot(v * 3.6, res_a * 1000, "o-")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_xlabel("speed (km/h)")
ax[1].set_ylabel("residual (mm)")
plt.tight_layout()
plt.savefig(here / "stopping_fit.png", dpi=150)
print("saved", here / "stopping_fit.png")