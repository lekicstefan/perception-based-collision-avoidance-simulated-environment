"""Runs the processor: handshake with Unity, then the processing cycle until END_OF_RUN. Or replays a recording.

  python -m server.run_processor                        # live, no run folder, nothing is written to disk
  python -m server.run_processor --run-dir              # live, run folder runs/<date_time>
  python -m server.run_processor --run-dir mytest       # live, run folder runs/mytest
  python -m server.run_processor --run-dir --no-record  # logs, but no python/stream.bin
  python -m server.run_processor --replay runs/<id> --speed 0                 # replay, no run folder
  python -m server.run_processor --replay runs/<id> --speed 0 --run-dir       # replay, runs/<date_time>_replay
  python -m server.run_processor --replay runs/<id> --run-dir mytest          # replay, runs/mytest
  python -m server.run_processor --pose-sigma-m 0.05 --pose-sigma-yaw-deg 0.5  # tell the tracker and the risk stage the pose noise of the simulation
  python -m server.run_processor --no-double-count                             # reachable radius without the present position sigma (7.9, off by default)
  python -m server.run_processor --pin 2,4                                     # pin this process to logical CPUs 2 and 4 and give it high priority (section 4.2)
  python -m server.run_processor --passing-speed-kmh 40                        # speed at which hazards that are only a precaution are passed (default 40, 0 = the occluded approach speed of the risk level)
  python -m server.run_processor --max-scans-per-cycle 0                       # no load shedding (default 2: at most the two newest scans per cycle, see shedding.py)
  python -m server.run_processor --loop --run-dir mytest                       # one run after another: after END_OF_RUN (or a lost link) wait for the next Unity run (restart or another scenario); runs/mytest, mytest_2, ...

Live: start this first, then press Play in Unity. With --run-dir the processor chooses the folder and tells Unity in
the READY message, so Unity writes its logs into the same folder. Without --run-dir neither side writes a run folder
(and there is no stream.bin, so no recording and no end-of-run boundary check).
"""
import argparse
import dataclasses
import math
import time

from server.communication.boundary import check_stream
from server.communication.transport import Link, LinkTimeout
from server.recording.replay import ReplayLink
from server.recording.run_log import RunLog
from server.recording.stream_log import StreamRecorder
from server.runtime.cycle import CycleRunner
from server.risk.cause_log import CauseLog
from server.risk.decision import DecisionConfig
from server.risk.hazards import HazardConfig
from server.risk.stage import StageConfig
from server.runtime.pinning import parse_cpu_list, pin_process
from server.runtime.pipeline_processor import PipelineProcessor, ProcessorConfig
from server.runtime.shedding import SheddingConfig


LOOP_PAUSE_S = 1.5      # --loop: pause between two runs. Unity sends END_OF_RUN five times over 0.4 s and then closes its sockets.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0, help="processing cycles per second (deadline = 1 / rate)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--run-dir", nargs="?", const="", default=None, metavar="NAME", help="write a run folder: runs/<NAME>, or runs/<date_time> when NAME is left out. Without this flag no run folder is created at all")
    ap.add_argument("--no-record", action="store_true", help="with --run-dir: do not write stream.bin")
    ap.add_argument("--replay", default=None, help="a stream.bin, a run folder or its python folder")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed, 1 = original timing, 0 = as fast as possible")
    ap.add_argument("--pose-sigma-m", type=float, default=0.0, help="pose noise of the car's position (m): goes to the tracker and to the distance sigma of the risk stage")
    ap.add_argument("--pose-sigma-yaw-deg", type=float, default=0.0, help="pose noise of the car's heading (degrees): goes to the tracker")
    ap.add_argument("--extent-across-m", type=float, default=ProcessorConfig().sigma_extent_across_m,
                    help="noise of the middle of the visible extent across the line of sight (m) for the tracker, 0 = off")
    ap.add_argument("--no-double-count", action="store_true", help="the reachable radius gets only the growth of the position sigma, not its present value (HazardConfig.subtract_current_sigma)")
    ap.add_argument("--pin", default=None, metavar="CPUS", help="pin this process to these logical CPUs and give it high priority, for example 2,4 or 2-5 (section 4.2: the processor and Unity run on separate cores in every timing experiment)")
    ap.add_argument("--max-scans-per-cycle", type=int, default=SheddingConfig().max_scans_per_cycle, metavar="N",
                    help="load shedding: when more scans are waiting, only the N newest are processed and the older ones are skipped and counted; 0 = never skip")
    ap.add_argument("--passing-speed-kmh", type=float, default=40.0, metavar="KMH",
                    help="speed at which the car passes a hazard that is only a precaution (a person beside the road, a hidden strip beside a parked car): its caps are never below it; 0 = the occluded approach speed of the risk level (the earlier behaviour)")
    ap.add_argument("--loop", action="store_true",
                    help="keep running: when a run ends (END_OF_RUN, a lost link) wait for the next run instead of exiting, so Unity can restart or switch scenarios. Each run gets its own run folder (NAME, NAME_2, ...). Stop with Ctrl+C. Not for --replay")
    args = ap.parse_args()
    if args.pose_sigma_m < 0.0 or args.pose_sigma_yaw_deg < 0.0:
        ap.error("the pose noise cannot be negative")
    if args.max_scans_per_cycle < 0:
        ap.error("--max-scans-per-cycle cannot be negative")
    if args.passing_speed_kmh < 0.0:
        ap.error("--passing-speed-kmh cannot be negative")
    if args.loop and args.replay is not None:
        ap.error("--loop does not work with --replay")
    pin_info = None
    if args.pin is not None:
        try:
            pin_info = pin_process(parse_cpu_list(args.pin))
        except (ValueError, OSError, NotImplementedError) as e:
            ap.error(f"--pin: {e}")
        print(f"pinned to logical CPUs {pin_info['cpus']} (mask 0x{pin_info['mask']:X}), priority {pin_info['priority']}")
    config = ProcessorConfig(sigma_extent_across_m=args.extent_across_m, pose_position_sigma_m=args.pose_sigma_m, pose_yaw_sigma_rad=math.radians(args.pose_sigma_yaw_deg),
                             stage=StageConfig(hazard=HazardConfig(subtract_current_sigma=args.no_double_count),
                                               decision=DecisionConfig(passing_speed_mps=args.passing_speed_kmh / 3.6)))
    print(f"pose noise: position {args.pose_sigma_m:g} m, heading {args.pose_sigma_yaw_deg:g} deg"
          f"{' (no pose noise told to the tracker)' if args.pose_sigma_m == 0.0 and args.pose_sigma_yaw_deg == 0.0 else ''}; "
          f"covariance in the reachable radius: {'growth only' if args.no_double_count else 'full (default)'}")
    print(f"passing speed for precaution hazards: {args.passing_speed_kmh:g} km/h" + (" (off: the occluded approach speed of the risk level)" if args.passing_speed_kmh == 0.0 else ""))
    runs = 0
    try:
        while True:
            runs += 1
            if args.loop:
                print(f"\n=================== run {runs} ===================")
            status = run_once(args, config, pin_info)
            if not args.loop or status == "interrupted":
                break
            time.sleep(LOOP_PAUSE_S)       # the old Unity run is still sending its last END_OF_RUN copies and closing its sockets
    except KeyboardInterrupt:
        print("interrupted")
    if args.loop:
        print(f"\n{runs} run(s) handled")


def run_once(args, config, pin_info) -> str:
    """One run: handshake, processing cycles until END_OF_RUN, reports, logs. Returns "ended", "timeout" (the link died) or "interrupted" (Ctrl+C).
    Everything of a run is created here (link, processor, run folder), so the next run starts from nothing."""
    replay = args.replay is not None
    logging_on = args.run_dir is not None

    link = ReplayLink(args.replay, speed=args.speed, period=1.0 / args.rate) if replay else Link(oracle=args.oracle)
    processor = PipelineProcessor(config)
    processor.telemetry_sink = getattr(link, "send_telemetry", None)          # the replay link has none: no telemetry then
    status = "ended"
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
                           run_dir=run_dir.as_posix() if run_dir else None, timeout=float("inf") if args.loop else 120.0)

        if run_dir is not None:
            run_log = RunLog(run_dir)
            run_log.write_meta(rate_hz=args.rate, oracle=args.oracle, run_dir=str(run_dir),
                               replay_of=str(link.path) if replay else None, replay_speed=args.speed if replay else None)
            if not replay and not args.no_record:
                recorder = StreamRecorder(run_log.dir / "stream.bin")
                link.attach_recorder(recorder)
            processor.log = CauseLog(run_log.dir / "cause.jsonl", keep=False)          # the cause of every cap and braking decision, for the evaluation
            print(f"logging to {run_log.dir}")
        else:
            print("no run folder (add --run-dir to log this run)")
        if not replay and recorder is None:
            link.stop_buffering()                            # nothing is recorded, drop what was buffered
        print(f"handshake done, calibration id {link.calibration_id}" if not replay else "replay started")
        runner = CycleRunner(link, processor, rate_hz=args.rate, run_log=run_log,
                             realtime=not (replay and args.speed <= 0), shedding=SheddingConfig(max_scans_per_cycle=args.max_scans_per_cycle))
        runner.run(args.seconds)
    except KeyboardInterrupt:
        print("interrupted")
        status = "interrupted"
    except LinkTimeout as e:
        print(f"LINK TIMEOUT: {e}")
        status = "timeout"
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
            if processor.log is not None:
                processor.log.close()
            run_log.write_summary(pin=pin_info, passing_speed_kmh=args.passing_speed_kmh, cycle=runner.stats() if runner else None,
                                  streams={n: dataclasses.asdict(s) for n, s in link.stats.items()},
                                  calibration_id=link.calibration_id, ended=link.ended, end_sim_time=link.end_sim_time,
                                  malformed_messages=link.error_count, first_errors=link.errors[:5],
                                  recorded_messages=recorder.count if recorder else None, boundary=boundary)
            run_log.close()
        link.close()
    return status


if __name__ == "__main__":
    main()