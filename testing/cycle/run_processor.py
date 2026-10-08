"""Runs the processor: handshake with Unity, then the processing cycle until END_OF_RUN. Or replays a recording.

  python server/run_processor.py                          # live, 30 cycles per second, until the run ends
  python server/run_processor.py --rate 15 --seconds 20
  python server/run_processor.py --no-record              # live, without writing python/stream.bin
  python server/run_processor.py --replay runs/<id>       # replay in original timing
  python server/run_processor.py --replay runs/<id> --speed 0   # replay as fast as possible (deterministic)

Live: start this first, then press Play in Unity. Logs go into <run folder>/python/ (the folder Unity announced in
HELLO), including stream.bin, the recording of everything received (about 1 to 2 MB per second of run).
Replay: logs go into a new runs/<time>_replay folder.
"""
import argparse
import dataclasses

from cycle import CycleRunner, Processor
from boundary import check_stream
from replay import ReplayLink
from run_log import RunLog
from stream_log import StreamRecorder
from transport import Link, LinkTimeout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0, help="processing cycles per second (deadline = 1 / rate)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="do not write stream.bin")
    ap.add_argument("--replay", default=None, help="a stream.bin, a run folder or its python folder")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = original timing, 0 = as fast as possible")
    args = ap.parse_args()
    replay = args.replay is not None

    link = ReplayLink(args.replay, speed=args.speed, period=1.0 / args.rate) if replay else Link(oracle=args.oracle)
    processor = Processor()
    run_log = recorder = runner = boundary = None
    try:
        if replay:
            print(f"replaying {link.path} ({'as fast as possible' if args.speed <= 0 else f'speed {args.speed:g}'})")
            link.handshake(warmup=processor.warmup)
            run_dir = str(RunLog.offline_dir("replay"))
        else:
            print("waiting for HELLO from Unity (press Play now) ...")
            link.handshake(warmup=processor.warmup, buffer_for_recording=not args.no_record)
            run_dir = link.hello.get("run_dir") or str(RunLog.offline_dir())
        run_log = RunLog(run_dir)
        run_log.write_meta(rate_hz=args.rate, oracle=args.oracle, run_dir=run_dir,
                           replay_of=str(link.path) if replay else None, replay_speed=args.speed if replay else None)
        if not replay:
            if args.no_record:
                link.stop_buffering()
            else:
                recorder = StreamRecorder(run_log.dir / "stream.bin")
                link.attach_recorder(recorder)
        print(f"handshake done, calibration id {link.calibration_id}" if not replay else "replay started")
        print(f"logging to {run_log.dir}")
        runner = CycleRunner(link, processor, rate_hz=args.rate, run_log=run_log, realtime=not (replay and args.speed <= 0))
        runner.run(args.seconds)
    except KeyboardInterrupt:
        print("interrupted")
    except LinkTimeout as e:
        print(f"LINK TIMEOUT: {e}")
    finally:
        if not replay:
            link.drain(200)
        if runner is not None:
            print("\n--- cycle report ---")
            print(runner.report())
        print("\n--- link report ---")
        print(link.summary())
        if recorder is not None:
            recorder.close()
            print(f"recorded {recorder.count} messages, {recorder.bytes / 1e6:.1f} MB, to {recorder.path}")
            rep = check_stream(recorder.path, normal_mode=not args.oracle)     # information-boundary check of the run
            boundary = {"ok": rep.ok, "messages": rep.messages, "counts": rep.counts, "problems": rep.problems[:10]}
            print("information boundary: " + ("OK" if rep.ok else f"{len(rep.problems)} PROBLEMS, first: {rep.problems[0]}"))
        if run_log is not None:
            run_log.write_summary(cycle=runner.stats() if runner else None,
                                  streams={n: dataclasses.asdict(s) for n, s in link.stats.items()},
                                  calibration_id=link.calibration_id, ended=link.ended, end_sim_time=link.end_sim_time,
                                  malformed_messages=link.error_count, first_errors=link.errors[:5],
                                  recorded_messages=recorder.count if recorder else None, boundary=boundary)
            run_log.close()
        link.close()


if __name__ == "__main__":
    main()