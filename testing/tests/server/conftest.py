"""Fixtures shared by all tests of the server package (applied automatically, no imports needed)."""
import itertools

import pytest

from server.communication.transport import Link
from testing.support.fake_unity import FakeUnity

_port = itertools.count(26000, 10)


@pytest.fixture
def ports():
    """A fresh set of six ports per test, so tests never collide."""
    base = next(_port)
    return {"session": base, "command": base + 1, "lidar": base + 2, "camera": base + 3, "pose": base + 4,
            "oracle": base + 5}


@pytest.fixture(autouse=True)
def cleanup(monkeypatch):
    """Close every Link and FakeUnity a test created, also when it fails (an open context blocks the exit)."""
    created = []
    orig_link, orig_fake = Link.__init__, FakeUnity.__init__

    def link_init(self, *a, **k):
        orig_link(self, *a, **k)
        created.append(self)

    def fake_init(self, *a, **k):
        orig_fake(self, *a, **k)
        created.append(self)

    monkeypatch.setattr(Link, "__init__", link_init)
    monkeypatch.setattr(FakeUnity, "__init__", fake_init)
    yield
    for obj in created:
        obj.close()