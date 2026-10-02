import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # drop warm-up frames
df = pd.read_csv(path)
n_all = len(df)
df = df[df.frame_id >= skip]
ok = df.dropna(subset=["t_sent_ms", "t_recv_ms", "t_applied_ms"])
print(f"frames: {n_all}, analysed: {len(ok)}, without complete echo: {len(df) - len(ok)}")

seg = {
    "enqueue->sent":   ok.t_sent_ms - ok.t_enq_ms,
    "sent->recv":      ok.t_recv_ms - ok.t_sent_ms,
    "recv->applied":   ok.t_applied_ms - ok.t_recv_ms,
    "total":           ok.t_applied_ms - ok.t_enq_ms,
}
print(f"\n{'segment (ms)':16s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
for name, s in seg.items():
    print(f"{name:16s} {s.mean():7.2f} {s.quantile(.5):7.2f} {s.quantile(.95):7.2f} "
          f"{s.quantile(.99):7.2f} {s.max():7.2f}")
tot = seg["total"]
print(f"\np99 - p50 of total = {tot.quantile(.99) - tot.quantile(.5):.2f} ms")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(tot, bins=80)
ax[0].set_xlabel("total loop delay (ms)")
ax[0].set_ylabel("count")
ax[1].plot(ok.t_enq_ms / 1000, tot, ".", ms=2)
ax[1].set_xlabel("run time (s)")
ax[1].set_ylabel("total loop delay (ms)")
plt.tight_layout()
out = path.replace(".csv", ".png")
plt.savefig(out, dpi=150)
print("saved", out)