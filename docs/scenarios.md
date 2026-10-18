# Scenarios

A scenario is **one Unity scene plus one JSON file**. The world (road, parked cars, people, lamp posts, buildings) is built by hand in the editor, one scene per scenario (similar scenarios start from a copy of the same environment). The JSON file in `configs/scenarios/<name>.json` holds the parameters: speeds, timing, fog, sensor configuration, end conditions. Scenario files are committed; run folders are not.

## How a scene uses a scenario

Each scenario scene has one `ScenarioSettings` component (`Assets/Scripts/Scenario/`). Its `Scenario` field names the file (`dev` for `configs/scenarios/dev.json`). On start it reads the file and sets the values on the components of the scene (before `SensorRig` and `Vehicle` wake up), then checks that the file belongs to the scene that is loaded. A bad scenario stops the run before the handshake (editor: leaves play mode; player: quits), with the reason in the log.

Precedence of a value, strongest first:

1. `--set path=value` on the command line (repeatable)
2. `Editor Overrides` on the component
3. the scenario file
4. the value set on the component in the scene (Inspector)

`--scenario NAME` replaces the `Scenario` field. `--sensors NAME` and `--seed N` (SensorRig) still win over the scenario's `sensors` group.

Every key is optional. A missing key leaves the scene's own value, so a scenario file lists only what it deliberately sets. What a run **used** is stored in full: `runs/<id>/config/scenario_resolved.json` has every setting, including the ones the file did not mention. `run.json` also records the scenario name, the file, the overrides and the system variant.

## Format (schema 1)

| key | meaning |
|---|---|
| `schema` | 1 |
| `name` | must equal the file name without `.json` |
| `description` | free text |
| `scene` | name of the Unity scene of this scenario (required, checked at start) |
| `systemVariant` | processor variant, `default` for now (phase 9); only recorded |
| `driver` | `route`, `cruiseSpeedMps`, `maxLateralAccelMps2`, `comfortDecelMps2`, `comfortAccelMps2`, `speedGain`, `driverMaxBrakeMps2`, `lookaheadMinM`, `lookaheadPerSpeedS`, `lookaheadMaxM`, `endMarginM` (the scripted driver, `RouteDriver`) |
| `vehicle` | `wheelbaseM`, `maxSteeringDeg`, `maxSteeringRateDegPerS`, `maxAccelMps2`, `maxBrakeMps2`, `actuationDelayS`, `jerkLimitMps3` (`Vehicle`) |
| `executor` | `capGain`, `enforceExpiry` (`CommandExecutor`) |
| `sensors` | `config` (file in `configs/sensors`), `seed` (base seed of every noise generator) |
| `environment` | `fogVisibilityM` (0 = clear) |
| `objects` | map: name of a scene object to `speedMps`, `startDelayS`, `loop`, `faceDirection` (a `WaypointMover`). Speed profiles and time-to-reach triggers are added in step 8.3 |
| `end` | `timeoutS` (default 120), `onCollision` (true), `onRouteFinished` (true), `stoppedSpeedMps` (0.1), `stoppedForS` (3, 0 = off). Enforced by the scenario manager from step 8.2; stored and checked now |

A misspelled key, an unknown object name, two moving objects with the same name, or an impossible value (for example a negative speed) is an error.

## Overrides and sweep points

    --set driver.cruiseSpeedMps=11.1
    --set objects.CrossingVehicle.startDelayS=4.2
    --set environment.fogVisibilityM=200

The path is the JSON path with dots. The value is read as JSON (number, `true`, `false`) or else as text. This is how the batch runner (phase 14) creates a sweep point from one scenario file. The overrides are listed in `run.json`.

## Notes

- The processor assumes the braking capability measured in phase 2 (8 m/s^2, 0.15 s effective delay). `vehicle.maxBrakeMps2` and `vehicle.actuationDelayS` are in the file so that a run documents the car it used; changing them changes the car, not the processor's model.
- The scenario does not set the processor's options (`--passing-speed-kmh`, `--pin`, pose noise sigmas). The batch runner passes those.
- `dev` is the reference file for the development scene. The core scenarios 1 to 8 are added in step 8.5.