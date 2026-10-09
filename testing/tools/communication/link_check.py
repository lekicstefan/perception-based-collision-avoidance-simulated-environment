"""Checks the link to a running Unity (or to fake_unity.py): handshake, streams, losses, commands.

Start this first, then press Play in Unity (or start the player). It prints a report when the run ends.

  python server/link_check.py                         # handshake, receive until END_OF_RUN or Ctrl+C
  python server/link_check.py --seconds 20            # stop after 20 s of data
  python server/link_check.py --stall-at 5 --stall-for 1.5   # freeze the processor once, see what is lost
  python server/link_check.py --command-demo          # after 3 s ask the car to hold 5 m/s (tests the command path)
  python server/link_check.py --oracle                # also listen to the oracle stream (Unity must be in oracle mode)
"""
import argparse
import time

from server.communication import protocol as P
from server.communication.transport import Link, LinkTimeout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds of data")
    ap.add_argument("--rate", type=float, default=30.0, help="processing cycles per second")
    ap.add_argument("--stall-at", type=float, default=None, help="seconds after the first data when the processor freezes")
    ap.add_argument("--stall-for", type=float, default=1.5, help="length of the freeze in seconds")
    ap.add_argument("--command-demo", action="store_true", help="send a speed-cap command from 3 s on")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    link = Link(oracle=args.oracle)
    cmd_id = 0
    worst_backlog = 0
    try:
        print("waiting for HELLO from Unity (press Play now) ...")
        link.handshake()
        print(f"handshake done, calibration id {link.calibration_id}")
        t0 = time.monotonic()
        stalled = False
        period = 1.0 / args.rate
        next_cycle = t0
        while not link.ended:
            elapsed = time.monotonic() - t0
            if args.seconds is not None and elapsed > args.seconds:
                print("time is up, stopping")
                break
            if args.stall_at is not None and not stalled and elapsed > args.stall_at:
                stalled = True
                print(f"[{elapsed:5.1f} s] processor frozen for {args.stall_for} s")
                time.sleep(args.stall_for)
            msgs = link.drain(0)
            worst_backlog = max(worst_backlog, len(msgs))
            link.check_alive()
            if args.command_demo and elapsed > 3.0:
                cmd_id += 1
                link.send_command(P.Command(id=cmd_id, data_timestamp=0.0, comfort_cap=5.0, safety_cap=8.0,
                                            comfort_decel=2.5, brake_request=0.0, validity=0.5, fallback_decel=0.0,
                                            processing_time=0.005, emergency=False, risk_level=0))
            next_cycle += period
            time.sleep(max(0.0, next_cycle - time.monotonic()))
    except KeyboardInterrupt:
        print("interrupted")
    except LinkTimeout as e:
        print(f"LINK TIMEOUT: {e}")
    finally:
        link.drain(200)
        print("\n--- report ---")
        print(link.summary())
        print(f"commands sent: {cmd_id}, largest number of messages drained in one cycle: {worst_backlog}")
        if link.errors:
            print("first malformed messages:", *link.errors[:5], sep="\n  ")
        link.close()


if __name__ == "__main__":
    main()