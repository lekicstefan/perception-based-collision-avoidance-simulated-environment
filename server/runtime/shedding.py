"""Load shedding: what the processor does when it is behind (section 4.2: late results are accepted, but the backlog must not grow).

The cycle drains everything that has arrived. When the processor is slower than the sensors, the next drain holds more than the last, the cycle
becomes longer, and the backlog grows without limit (observed in the second live smoke test: cycles of 1 s with 10 scans each, commands about 1 s old, 76 % of them expired in
Unity). A real car does not work through old data. This module bounds the backlog:

    scans    at most `max_scans_per_cycle` LiDAR scans are processed per cycle, the NEWEST ones; the older ones are skipped (the tracker is timestamp-driven:
             it predicts through the gap, with a growing uncertainty)
    camera   camera frames older than the oldest scan that is kept are skipped with the scans before them: they can no longer take part in a
             refinement or a bearing update that matters
    pose     never skipped: cheap, and the tracker needs the pose at the capture time of every scan that is kept

In normal operation a cycle holds zero or one scan, so nothing is skipped; a hiccup of the operating system that delays one cycle by 100 to 200 ms holds two. Every skipped
message is counted (cycles.csv, report, summary.json): the number of skipped scans is a result, not a secret. `max_scans_per_cycle = 0` switches shedding off.
"""
from __future__ import annotations

from dataclasses import dataclass

from server.communication import protocol as P


@dataclass(frozen=True)
class SheddingConfig:
    max_scans_per_cycle: int = 2        # 0: never skip anything


def shed(msgs: list, cfg: SheddingConfig) -> tuple[list, int, int]:
    """msgs sorted by timestamp. Returns (the messages to process, scans skipped, camera frames skipped)."""
    n = cfg.max_scans_per_cycle
    if n <= 0:
        return msgs, 0, 0
    scan_times = [m.header.t for m in msgs if m.header.type == P.MsgType.LIDAR]
    if len(scan_times) <= n:
        return msgs, 0, 0
    keep_from = scan_times[-n]                  # capture time of the oldest scan that is kept
    kept, scans, cameras = [], 0, 0
    for m in msgs:
        typ, t = m.header.type, m.header.t
        if typ == P.MsgType.LIDAR and t < keep_from:
            scans += 1
        elif typ == P.MsgType.CAMERA and t < keep_from:
            cameras += 1
        else:
            kept.append(m)
    return kept, scans, cameras