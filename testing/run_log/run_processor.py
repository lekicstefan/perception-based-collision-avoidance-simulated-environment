"""Runs the processor: handshake with Unity, then the processing cycle until END_OF_RUN.

  python server/run_processor.py                      # 30 cycles per second, until the run ends
  python server/run_processor.py --rate 15 --seconds 20
  python server/run_processor.py --oracle             # also listen to the oracle stream (Unity in oracle mode)

Start this first, then press Play in Unity. Logs go into <run folder>/python/ (the folder Unity announced in HELLO).
For now the processor does nothing with the data (the pipeline of phases 5 to 7 plugs in through the Processor class
in cycle.py); this run shows the timing of the cycle itself.
"""
import argparse
import dataclasses

from cycle import CycleRunner, Processor
from run_log import RunLog
from transport import Link, LinkTimeout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0, help="processing cycles per second (deadline = 1 / rate)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    link = Link(oracle=args.oracle)
    processor = Processor()
    run_log = None
    runner = None
    try:
        print("waiting for HELLO from Unity (press Play now) ...")
        link.handshake(warmup=processor.warmup)
        run_dir = link.hello.get("run_dir") or str(RunLog.offline_dir())
        run_log = RunLog(run_dir)
        run_log.write_meta(rate_hz=args.rate, oracle=args.oracle, run_dir=run_dir)
        print(f"handshake done, calibration id {link.calibration_id}")
        print(f"logging to {run_log.dir}")
        runner = CycleRunner(link, processor, rate_hz=args.rate, run_log=run_log)
        runner.run(args.seconds)
    except KeyboardInterrupt:
        print("interrupted")
    except LinkTimeout as e:
        print(f"LINK TIMEOUT: {e}")
    finally:
        link.drain(200)
        if runner is not None:
            print("\n--- cycle report ---")
            print(runner.report())
        print("\n--- link report ---")
        print(link.summary())
        if run_log is not None:
            run_log.write_summary(cycle=runner.stats() if runner else None,
                                  streams={n: dataclasses.asdict(s) for n, s in link.stats.items()},
                                  calibration_id=link.calibration_id, ended=link.ended, end_sim_time=link.end_sim_time,
                                  malformed_messages=link.error_count, first_errors=link.errors[:5])
            run_log.close()
        link.close()


if __name__ == "__main__":
    main()