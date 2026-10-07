# Run folder

Every run writes into one folder, `runs/<run id>/` (not committed, listed in `.gitignore`). Unity creates it (run id = date and time, or `--run-id`; root = `runs/` next to `configs/`, or `--runs-root`) and sends its absolute path to the processor in the HELLO message (`run_dir`). The processor only uses it to know where to write its logs and never parses it for anything else.

```
runs/<run id>/
  run.json                 Unity: run id, scenario, seed, versions, hardware, command line, and at the end the end reason, impact speed, frame statistics, sim-to-real time ratio
  config/                  copies of the configuration used (sensors_resolved.json, calibration.json, later scenario and system variant)
  unity/                   ground truth (ego.csv, hazards.csv, visibility.csv, hazard_info.csv), commands.csv (what the executor applied each physics step), frames.csv (frame times)
  python/                  meta.json (git commit, dirty flag, Python and package versions, arguments), cycles.csv (per-cycle timings), commands.csv (every command sent), summary.json (timing percentiles, per-stream losses), later tracks.csv, risk.csv, cause log, stream/
```

Tables are CSV because they are written row by row and survive a crash. They are converted to Parquet when the data is analysed (phase 16). The commit hash is recorded by the processor side (`python/meta.json`), which runs from the same checkout as the Unity project.