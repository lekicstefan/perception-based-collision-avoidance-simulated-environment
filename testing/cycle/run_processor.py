"""Runs the processor: handshake with Unity, then the processing cycle until END_OF_RUN.

  python server/run_processor.py                      # 30 cycles per second, until the run ends
  python server/run_processor.py --rate 15 --seconds 20
  python server/run_processor.py --oracle             # also listen to the oracle stream (Unity in oracle mode)

Start this first, then press Play in Unity. For now the processor does nothing with the data (the pipeline of
phases 5 to 7 plugs in through the Processor class in cycle.py); this run shows the timing of the cycle itself.
"""
import argparse

from cycle import CycleRunner, Processor
from transport import Link, LinkTimeout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0, help="processing cycles per second (deadline = 1 / rate)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    link = Link(oracle=args.oracle)
    processor = Processor()
    runner = CycleRunner(link, processor, rate_hz=args.rate)
    try:
        print("waiting for HELLO from Unity (press Play now) ...")
        link.handshake(warmup=processor.warmup)
        print(f"handshake done, calibration id {link.calibration_id}; running at {args.rate:g} cycles per second")
        runner.run(args.seconds)
    except KeyboardInterrupt:
        print("interrupted")
    except LinkTimeout as e:
        print(f"LINK TIMEOUT: {e}")
    finally:
        print("\n--- cycle report ---")
        print(runner.report())
        print("\n--- link report ---")
        print(link.summary())
        link.close()


if __name__ == "__main__":
    main()