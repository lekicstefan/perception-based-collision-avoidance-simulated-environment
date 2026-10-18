# Run folder

Every run writes into one folder, `runs/<run id>/` (not committed, listed in `.gitignore`).

**Who decides.** The processor. `python -m server.run_processor --run-dir [name]` creates `runs/<name>/` (the name defaults to the date and time; a replay adds a label) and sends its absolute path to Unity in the READY message (`run_dir`, see `docs/protocol.md`). Unity creates the folder if it does not exist yet and writes into it. Without `--run-dir` no run folder exists, and the Unity loggers (run log, ground truth) switch themselves off. The run id is the name of the folder.

```
runs/<run id>/
  run.json                 Unity: run id, scenario, seed, versions, hardware, command line (only the executable, our own --arguments and a few harmless Unity
                           arguments; the rest is counted, so the file never contains a session token), and at the end the end reason, impact speed, frame
                           statistics, sim-to-real time ratio
  config/                  copies of the configuration used (sensors_resolved.json, calibration.json; later scenario and system variant)
  unity/                   ground truth (ego.csv, hazards.csv, visibility.csv, hazard_info.csv; written by default whenever there is a run folder),
                           commands.csv (what the executor applied each physics step), frames.csv (frame times)
  python/                  meta.json (git commit, dirty flag, Python and package versions, arguments)
                           cycles.csv (per cycle: arrival counts, timings, overrun flag, dropped_lidar and dropped_camera from the load shedding)
                           commands.csv (every command sent)
                           cause.jsonl (the cause of every cap and braking decision, with floor, velocity and track age; one line per scan)
                           summary.json (timing percentiles, per-stream losses, load shedding totals, CPU pinning, passing speed)
                           stream.bin (the recorded sensor stream: 8-byte magic "AVREC001", then records of float64 arrival seconds, uint32 length, raw message bytes)
                           later tracks.csv, risk.csv
```

Tables are CSV because they are written row by row and survive a crash. They are converted to Parquet when the data is analysed (phase 16). The commit hash is recorded by the processor side (`python/meta.json`), which runs from the same checkout as the Unity project.