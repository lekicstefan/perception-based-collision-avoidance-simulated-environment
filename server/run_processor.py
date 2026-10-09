"""Runs the processor: handshake with Unity, then the processing cycle until END_OF_RUN. Or replays a recording.

  python -m server.run_processor                        # live, no run folder, nothing is written to disk
  python -m server.run_processor --run-dir              # live, run folder runs/<date_time>
  python -m server.run_processor --run-dir mytest       # live, run folder runs/mytest
  python -m server.run_processor --run-dir --no-record  # logs, but no python/stream.bin
  python -m server.run_processor --replay runs/<id> --speed 0                 # replay, no run folder
  python -m server.run_processor --replay runs/<id> --speed 0 --run-dir       # replay, runs/<date_time>_replay
  python -m server.run_processor --replay runs/<id> --run-dir mytest          # replay, runs/mytest

Live: start this first, then press Play in Unity. With --run-dir the processor chooses the folder and tells Unity in
the READY message, so Unity writes its logs into the same folder. Without --run-dir neither side writes a run folder
(and there is no stream.bin, so no recording and no end-of-run boundary check).
"""
import argparse
import dataclasses

from server.communication.boundary import check_stream
from server.communication.transport import Link, LinkTimeout
from server.recording.replay import ReplayLink
from server.recording.run_log import RunLog
from server.recording.stream_log import StreamRecorder
from server.runtime.cycle import CycleRunner, Processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0, help="processing cycles per second (deadline = 1 / rate)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--run-dir", nargs="?", const="", default=None, metavar="NAME", help="write a run folder: runs/<NAME>, or runs/<date_time> when NAME is left out. Without this flag no run folder is created at all")
    ap.add_argument("--no-record", action="store_true", help="with --run-dir: do not write stream.bin")
    ap.add_argument("--replay", default=None, help="a stream.bin, a run folder or its python folder")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = original timing, 0 = as fast as possible")
    args = ap.parse_args()
    replay = args.replay is not None
    logging_on = args.run_dir is not None

    link = ReplayLink(args.replay, speed=args.speed, period=1.0 / args.rate) if replay else Link(oracle=args.oracle)
    processor = Processor()
    run_log = recorder = runner = boundary = None
    try:
        run_dir = None
        if replay:
            print(f"replaying {link.path} ({'as fast as possible' if args.speed <= 0 else f'speed {args.speed:g}'})")
            link.handshake(warmup=processor.warmup)
            if logging_on:
                run_dir = RunLog.new_run_dir(args.run_dir, label="replay")
        else:
            if logging_on:
                run_dir = RunLog.new_run_dir(args.run_dir)          # chosen now, sent to Unity in READY, created below
            print("waiting for HELLO from Unity (press Play now) ...")
            link.handshake(warmup=processor.warmup, buffer_for_recording=logging_on and not args.no_record,
                           run_dir=run_dir.as_posix() if run_dir else None)

        if run_dir is not None:
            run_log = RunLog(run_dir)
            run_log.write_meta(rate_hz=args.rate, oracle=args.oracle, run_dir=str(run_dir),
                               replay_of=str(link.path) if replay else None, replay_speed=args.speed if replay else None)
            if not replay and not args.no_record:
                recorder = StreamRecorder(run_log.dir / "stream.bin")
                link.attach_recorder(recorder)
            print(f"logging to {run_log.dir}")
        else:
            print("no run folder (add --run-dir to log this run)")
        if not replay and recorder is None:
            link.stop_buffering()                            # nothing is recorded, drop what was buffered
        print(f"handshake done, calibration id {link.calibration_id}" if not replay else "replay started")
        runner = CycleRunner(link, processor, rate_hz=args.rate, run_log=run_log,
                             realtime=not (replay and args.speed <= 0))
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