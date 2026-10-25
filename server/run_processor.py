import argparse
import math
import time

from server.communication import protocol as P
from server.communication.transport import Link, LinkTimeout
from server.recording.run_log import RunLog
from server.runtime.pinning import parse_cpu_list, pin_process

LOOP_PAUSE_S = 1.5      # Unity sends END_OF_RUN five times over 0.4 s and then closes its sockets
MIN_BRAKE = 1.0         # m/s^2, a risk needing less braking than this is not sent yet
COLUMNS = ["t_ms", "capture_ns", "built_ns", "sent_ns", "recv_ns", "parsed_ns", "done_ns", "cmd_sent_ns", "bytes"]


class MessageBook:
    """The processor's list of active messages, one per object id. An update only moves a stop position closer."""

    def __init__(self):
        self.stops: dict[int, float] = {}

    def offer(self, object_id: int, stop: float, speed: float, odometer: float) -> None:
        """A risk seen in this frame. A new one becomes a message only when the braking it needs reaches MIN_BRAKE."""
        if object_id in self.stops:
            self.stops[object_id] = min(self.stops[object_id], stop)
            return
        remaining = stop - odometer
        if remaining <= 0.0 or speed * speed / (2.0 * remaining) >= MIN_BRAKE:
            self.stops[object_id] = stop

    def remove(self, object_id: int) -> None:
        self.stops.pop(object_id, None)

    def nearest(self) -> P.StopMessage | None:
        if not self.stops:
            return None
        oid = min(self.stops, key=self.stops.get)
        return P.StopMessage(oid, self.stops[oid])


class DriveScript:
    def __init__(self):
        self.book = MessageBook()
        self.risks: dict[int, float] = {}
        self.fired = set()

    def once(self, name: str, t: float, at: float) -> bool:
        if t >= at and name not in self.fired:
            self.fired.add(name)
            return True
        return False

    def gone(self, object_id: int) -> None:
        self.risks.pop(object_id, None)
        self.book.remove(object_id)

    def command(self, h: P.Header, odometer: float) -> P.Command:
        t, risks = h.t, self.risks
        if self.once("add1", t, 2.0):
            risks[1] = odometer + 110.0
        if self.once("add2", t, 5.0):
            risks[2] = risks[1] + 10.0
        if self.once("rm2", t, 18.0):
            self.gone(2)
        if self.once("rm1", t, 20.0):
            self.gone(1)
        if self.once("add3", t, 24.0):
            risks[3] = odometer + 45.0
        if self.once("tighten3", t, 26.0):
            risks[3] -= 8.0
        if self.once("loosen3", t, 28.0):
            risks[3] += 5.0
        if self.once("rm3", t, 34.0):
            self.gone(3)
        for oid, stop in risks.items():
            self.book.offer(oid, stop, h.speed, odometer)
        return P.Command(data_timestamp=t, message=self.book.nearest())


def busy(ms: float) -> None:
    end = time.perf_counter_ns() + int(ms * 1e6)
    while time.perf_counter_ns() < end:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="run folder runs/<name>, default runs/<date_time>")
    ap.add_argument("--load-ms", type=float, default=0.0, help="busy work per bundle, stands in for the processing")
    ap.add_argument("--script", choices=["none", "drive"], default="none", help="drive: scripted objects for the drive_test scenario")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--pin", default=None, metavar="CPUS", help="logical CPUs to pin this process to, for example 2,4 or 2-5")
    ap.add_argument("--loop", action="store_true", help="after a run wait for the next one (the name gets _2, _3 ... when it exists)")
    args = ap.parse_args()
    if args.pin is not None:
        info = pin_process(parse_cpu_list(args.pin))
        print(f"pinned to logical CPUs {info['cpus']}, priority {info['priority']}")
    runs = 0
    try:
        while True:
            runs += 1
            status = run_once(args)
            if not args.loop or status == "interrupted":
                break
            time.sleep(LOOP_PAUSE_S)
    except KeyboardInterrupt:
        print("interrupted")


def run_once(args) -> str:
    link = Link()
    status = "ended"
    log = None
    try:
        run_dir = RunLog.new_run_dir(args.name)
        print("waiting for HELLO from Unity (press Play now) ...")
        link.handshake(run_dir=run_dir.as_posix(), timeout=float("inf") if args.loop else 120.0)
        run_log = RunLog(run_dir)
        run_log.write_meta(load_ms=args.load_ms, pin=args.pin, script=args.script)
        log = run_log.table("latency", COLUMNS)
        print(f"handshake done, calibration id {link.calibration_id}, logging to {run_log.dir}")
        script = DriveScript() if args.script == "drive" else None
        t0 = time.monotonic()
        while not link.ended:
            link.check_alive()
            for m in link.receive(50):
                if m.header.type != P.MsgType.BUNDLE:
                    continue
                b = P.parse_bundle(m.raw, m.header)
                parsed = time.perf_counter_ns()
                busy(args.load_ms)
                done = time.perf_counter_ns()
                link.send_command(script.command(m.header, b.odometer) if script else P.Command(data_timestamp=m.header.t))
                sent = time.perf_counter_ns()
                log.row(round(m.header.t * 1000), b.capture_ns, b.built_ns, b.sent_ns, m.recv_ns, parsed, done, sent, len(m.raw))
            if args.seconds is not None and time.monotonic() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        print("interrupted")
        status = "interrupted"
    except LinkTimeout as e:
        print(f"LINK TIMEOUT: {e}")
        status = "timeout"
    finally:
        link.receive(200)
        print(link.summary())
        if log is not None:
            run_log.close()
            print(f"{log.rows} bundles answered. Report: python -m server.latency_report {run_dir.as_posix()}")
        link.close()
    return status


if __name__ == "__main__":
    main()