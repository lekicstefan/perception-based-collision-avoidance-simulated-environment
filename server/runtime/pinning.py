from __future__ import annotations

import ctypes
import os
import sys

HIGH_PRIORITY_CLASS = 0x00000080


def parse_cpu_list(text: str) -> list[int]:
    """'2,4' -> [2, 4]; '2-5' -> [2, 3, 4, 5]; '0-1,6' -> [0, 1, 6]. Raises ValueError for anything else."""
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            a = int(lo)
            b = int(hi) if sep else a
        except ValueError:
            raise ValueError(f"cannot read '{part}' as a CPU number or a range (use for example 2,4 or 2-5)") from None
        if a < 0 or b < a:
            raise ValueError(f"'{part}' is not a valid CPU number or range")
        cpus.update(range(a, b + 1))
    if not cpus:
        raise ValueError("no CPU given")
    return sorted(cpus)


def cpu_mask(cpus) -> int:
    """The affinity bit mask of a list of logical CPUs (CPU 0 is bit 0)."""
    mask = 0
    for c in cpus:
        mask |= 1 << int(c)
    return mask


def pin_process(cpus: list[int], high_priority: bool = True) -> dict:
    """Pin this process to the given logical CPUs and (optionally) raise its priority. Returns what was applied:
    {"cpus", "mask", "priority", "platform"}. Raises ValueError for a CPU that does not exist, OSError when the system refuses,
    NotImplementedError where pinning is not supported."""
    if sys.platform == "win32":
        return _pin_windows(list(cpus), high_priority)
    if hasattr(os, "sched_setaffinity"):
        return _pin_linux(list(cpus), high_priority)
    raise NotImplementedError(f"CPU pinning is not supported on {sys.platform}")


def _pin_windows(cpus: list[int], high_priority: bool) -> dict:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = ctypes.c_int
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = ctypes.c_int
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.SetPriorityClass.restype = ctypes.c_int
    handle = k.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(0), ctypes.c_size_t(0)
    if not k.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    mask = cpu_mask(cpus)
    missing = [c for c in cpus if not (system_mask.value >> c) & 1]
    if missing:
        raise ValueError(f"CPU {missing} does not exist on this machine (available logical CPUs: 0 to {system_mask.value.bit_length() - 1})")
    if not k.SetProcessAffinityMask(handle, mask):
        raise ctypes.WinError(ctypes.get_last_error())
    priority = "unchanged"
    if high_priority:
        if not k.SetPriorityClass(handle, HIGH_PRIORITY_CLASS):
            raise ctypes.WinError(ctypes.get_last_error())
        priority = "high"
    return {"cpus": cpus, "mask": mask, "priority": priority, "platform": sys.platform}


def _pin_linux(cpus: list[int], high_priority: bool) -> dict:
    allowed = os.sched_getaffinity(0)
    missing = [c for c in cpus if c not in allowed]
    if missing:
        raise ValueError(f"CPU {missing} is not available to this process (available: {sorted(allowed)})")
    os.sched_setaffinity(0, set(cpus))
    priority = "unchanged"
    if high_priority:
        try:
            os.setpriority(os.PRIO_PROCESS, 0, -10)
            priority = "high"
        except OSError:
            priority = "unchanged (raising the priority needs more rights)"
    return {"cpus": cpus, "mask": cpu_mask(cpus), "priority": priority, "platform": sys.platform}