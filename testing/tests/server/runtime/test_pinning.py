"""CPU pinning (server/runtime/pinning.py): the parsing and the mask are checked in this process; the real system calls are made in a child process,
so that the test session itself is never pinned."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from server.runtime.pinning import cpu_mask, parse_cpu_list

ROOT = Path(__file__).resolve().parents[4]


def test_a_list_and_a_range_are_read():
    assert parse_cpu_list("2,4") == [2, 4]
    assert parse_cpu_list("2-5") == [2, 3, 4, 5]
    assert parse_cpu_list("0-1, 6") == [0, 1, 6]
    assert parse_cpu_list("4,2,4") == [2, 4]                       # sorted, no duplicates


@pytest.mark.parametrize("text", ["", " ", "a", "2-", "-1", "5-2", "2,x", "1.5"])
def test_nonsense_is_refused(text):
    with pytest.raises(ValueError):
        parse_cpu_list(text)


def test_the_mask_has_one_bit_per_cpu():
    assert cpu_mask([2, 4]) == 0b10100
    assert cpu_mask([0]) == 1
    assert cpu_mask(range(4)) == 0b1111


def test_a_process_can_be_pinned_to_cpu_0():
    code = "import json; from server.runtime.pinning import pin_process; print(json.dumps(pin_process([0], high_priority=True)))"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["cpus"] == [0] and info["mask"] == 1


def test_a_cpu_that_does_not_exist_is_refused():
    code = "from server.runtime.pinning import pin_process; pin_process([4000])"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "ValueError" in r.stderr