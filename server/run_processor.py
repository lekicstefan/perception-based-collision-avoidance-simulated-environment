import argparse
import time

from server.communication import protocol as P
from server.communication.transport import Link, LinkTimeout
from server.recording.run_log import RunLog
from server.runtime.pinning import parse_cpu_list, pin_process

LOOP_PAUSE_S = 1.5      # Unity sends END_OF_RUN five times over 0.4 s and then closes its sockets
COLUMNS = ["frame", "capture_ns", "built_ns", "sent_ns", "recv_ns", "parsed_ns", "done_ns", "cmd_sent_ns", "bytes"]
NO_LIMITS = dict(comfort_cap=100.0, safety_cap=100.0, comfort_decel=2.0, brake_request=0.0, fallback_decel=0.0, emergency=False, risk_level=0)


def busy(ms: float) -> None:
    end = time.perf_counter_ns() + int(ms * 1e6)
    while time.perf_counter_ns() < end:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="run folder runs/<name>, default runs/<date_time>")
    ap.add_argument("--load-ms", type=float, default=0.0, help="busy work per bundle, stands in for the processing")
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
        run_log.write_meta(load_ms=args.load_ms, pin=args.pin)
        log = run_log.table("latency", COLUMNS)
        print(f"handshake done, calibration id {link.calibration_id}, logging to {run_log.dir}")
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
                link.send_command(P.Command(id=b.frame_id, data_timestamp=m.header.t, validity=0.5,
                                            processing_time=(done - parsed) / 1e9, **NO_LIMITS))
                sent = time.perf_counter_ns()
                log.row(b.frame_id, b.capture_ns, b.built_ns, b.sent_ns, m.recv_ns, parsed, done, sent, len(m.raw))
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