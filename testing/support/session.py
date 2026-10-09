"""A fake Unity running in a thread next to a real Link (shared by the communication and recording tests)."""
import threading
import time

from testing.support.fake_unity import FakeUnity


class Session:
    """Runs the fake Unity in a thread: wait for READY, then run `script(fake)`."""

    def __init__(self, ports, script=None, oracle=False, calibration=None):
        self.fake = FakeUnity(oracle=oracle, ports=ports, calibration=calibration)
        self.script = script
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.fake.wait_ready()
            if self.script:
                self.script(self.fake)
        except Exception as e:      # reported by the test through .error
            self.error = e

    def finish(self):
        self.thread.join(10)
        assert self.error is None, self.error
        self.fake.close()


def first_messages(fake):
    """Three poses, two LiDAR scans and two camera frames, right after the handshake."""
    for i in range(3):
        fake.send_pose(0.02 * i)
    for i in range(2):
        fake.send_lidar(0.1 * i)
        fake.send_camera(0.033 * i)


def drain_until(link, done, timeout=5.0):
    """Keep draining until done() is true (or the timeout). Returns everything received.

    Messages on different sockets arrive independently, so one drain() call can return before all of them are here."""
    msgs = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        msgs += link.drain(50)
        if done():
            break
    return msgs