from __future__ import annotations

import struct
import time
from pathlib import Path

MAGIC = b"AVREC001"
RECORD = struct.Struct("<dI")


class StreamRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "wb")
        self.f.write(MAGIC)
        self.t0: float | None = None
        self.count = 0
        self.bytes = 0
        self._last_flush = time.monotonic()

    def write(self, raw: bytes, arrival: float | None = None) -> None:
        """arrival: time.monotonic() value of the moment the message arrived (default: now)."""
        if arrival is None:
            arrival = time.monotonic()
        if self.t0 is None:
            self.t0 = arrival
        self.f.write(RECORD.pack(max(0.0, arrival - self.t0), len(raw)))
        self.f.write(raw)
        self.count += 1
        self.bytes += len(raw)
        if time.monotonic() - self._last_flush > 1.0:
            self.f.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()


def resolve_stream_path(path) -> Path:
    """Accepts stream.bin itself, a run folder, or a run's python folder."""
    p = Path(path)
    if p.is_dir():
        for cand in (p / "python" / "stream.bin", p / "stream.bin"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"no stream.bin in {p}")
    return p


def iter_records(path):
    """Yields (arrival seconds, raw message) one record at a time, so long recordings need little memory."""
    with open(resolve_stream_path(path), "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a stream recording (bad magic)")
        while True:
            head = f.read(RECORD.size)
            if len(head) < RECORD.size:
                return
            arrival, n = RECORD.unpack(head)
            raw = f.read(n)
            if len(raw) < n:                 # cut off by a crash
                return
            yield arrival, raw