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
| `objects` | map: name of a scene object to its motion script (a `WaypointMover`), see below |
| `end` | `timeoutS` (default 120), `onCollision` (true), `onRouteFinished` (true), `stoppedSpeedMps` (0.1), `stoppedForS` (3, 0 = off) |

A misspelled key, an unknown object name, two moving objects with the same name, or an impossible value (for example a negative speed) is an error.

## Moving objects

A moving object is a scene object with a `WaypointMover` component. It follows the waypoints (the children of its `Path` object, in Hierarchy order) in plan view and keeps its height. Per object, in `objects.<name>`:

| key | meaning |
|---|---|
| `speedMps` | speed when it starts to move |
| `startDelayS` | seconds after the run begins (not together with `trigger`) |
| `loop`, `faceDirection` | as in the component |
| `trigger` | start so that the object meets the ego car at a chosen time offset (below) |
| `events` | list of speed changes (below). A list in the file replaces the one set in the scene |

### Speed changes (`events`)

Each event gives the object a new target speed. Exactly one key says when it happens:

| key | the event happens ... |
|---|---|
| `atTimeS` | when the object has been moving for this many seconds |
| `atPathM` | when it has travelled this many metres along its path |
| `afterPreviousS` | this many seconds after the previous event (after the start, for the first one) |

`speedMps` is the new speed (required). `accelMps2` is how fast the speed changes toward it; missing or 0 means **at once**, a sudden change. Events fire in list order, each one after the one before it. A pedestrian who stops half way and then runs:

    "events": [
      { "atPathM": 3.0, "speedMps": 0.0 },
      { "afterPreviousS": 1.5, "speedMps": 4.0, "accelMps2": 6.0 }
    ]

A vehicle that brakes hard: `{ "atTimeS": 6.0, "speedMps": 0.0, "accelMps2": 6.0 }`. A lane change is not an event: the lane change is the shape of the path (waypoints), the object keeps its speed.

The timing is on the object's own clock, so it does not depend on the ego car. Events are tuning points for sweeps: `--set objects.Ped.events.0.atTimeS=2.5` (a number in the path is the position in the list, counted from 0; the list must exist in the file or the scene).

### Time-to-reach trigger (`trigger`)

A timed start relative to the ego car: the object starts so that it reaches the **conflict point** a chosen number of seconds after the **front of the ego car** does. Sweeping the offset gives comparable situations whatever the scripted speed of the car.

    "trigger": { "arrivalOffsetS": 0.0 }

- `arrivalOffsetS` (required): object arrival minus ego arrival. 0 = both at the conflict point at the same moment (a collision course if the object is in the lane then). Positive: the ego car is there first. Negative: the object is there first.
- The conflict point is a marker object in the scene (an empty GameObject, assigned to `Conflict Point` on the mover) placed where the object's path meets the ego route. The distance along the route and along the object's path are found by projecting the marker on both. Instead of a marker, both distances can be given: `egoRouteM` and `objectPathM`.
- The reference of the ego car is its front bumper (the front of its collision box). The reference of the object is its pivot, so put the pivot of a vehicle where the front or the centre should count.
- The start time is computed **before the run**, in the frozen first frames, from the scripted driver: the speed profile, its P controller, and the actuation delay, jerk limit and limits of the vehicle (`EgoPrediction`). It is the schedule of a car that nobody intervenes with, so it is the same for every system variant and for the no-safety baseline. It is not re-computed from the real car, so a system that slows the car does not change when the pedestrian starts. A start time that would be before the run begins stops the run with an explanation (move the object further back, lengthen the approach or use a larger offset).
- How well the schedule held is in `run.json`, `end.outcome.objects.<name>.trigger`: predicted and achieved arrival times and the achieved offset. The achieved ego arrival only exists when the car got to the conflict point, and it is only close to the prediction when nothing slowed the car before it.

Limits: the prediction ignores steering (the route arc length is used as the distance driven), so it is a little off on tight bends; the conflict point must be before the point where the driver stops (route end minus `endMarginM`); the ego car starts at rest.

## Overrides and sweep points

    --set driver.cruiseSpeedMps=11.1
    --set objects.CrossingVehicle.trigger.arrivalOffsetS=-0.5
    --set objects.Ped.events.0.atTimeS=2.5
    --set environment.fogVisibilityM=200

The path is the JSON path with dots. The value is read as JSON (number, `true`, `false`) or else as text. This is how the batch runner (phase 14) creates a sweep point from one scenario file. The overrides are listed in `run.json`.

## Running a scenario: run lifecycle and controls

Every scenario scene has a `ScenarioManager`. A run starts when the processor answers READY. It ends at the first of: collision (the car is frozen at the impact), route finished (the driver reached the end and the car stands still), stopped (the car drove and then stood still for `end.stoppedForS`), timeout (`end.timeoutS` of simulation time), or manual (restart, End run). The manager then tells the processor (END_OF_RUN), writes the outcome to `run.json` (`end.outcome`: reason, sim time, final speed, route progress, impact speed and hazard, the report of every moving object, and whether the run is valid for experiments), waits until the logs are closed and freezes the simulation.

Run the processor with `--loop` so that it waits for the next run: `python -m server.run_processor --pin 2,4 --run-dir name --loop`. Each run gets its own folder (`name`, `name_2`, ...). Unity reloads the scene on restart (the scene must be in the Build Settings). `--quit-on-end` (Unity command line) quits the player when the run has ended, for batch runs.

Keys (click the Game view first): `R` restart, `P` pause / resume, `X` end the run, `+` / `-` speed 0.25x to 4x, `0` back to 1x. A run at any speed other than 1x is **not valid for experiments** (both programs run on real wall-clock time) and `run.json` says so; every run starts at 1x. Pauses are recorded (`pauses`, `pausedRealS`) and do not invalidate a run.

Views (`ViewControls`, debug only, never sent to the processor): `1`-`9` / `C` camera, `D` lidar dots, `B` lidar beams (`,` `.` move the ring), `L` lidar view (range image / bird's-eye), `V` camera view, `G` hazard boxes, `T` route, `H` hide the HUD. The choices are kept across restarts.

## Notes

- The processor assumes the braking capability measured in phase 2 (8 m/s^2, 0.15 s effective delay). `vehicle.maxBrakeMps2` and `vehicle.actuationDelayS` are in the file so that a run documents the car it used; changing them changes the car, not the processor's model.
- The scenario does not set the processor's options (`--passing-speed-kmh`, `--pin`, pose noise sigmas). The batch runner passes those.
- `dev` is the reference file for the development scene, `dev_trigger` tests the triggers and speed events there. The core scenarios 1 to 8 are added in step 8.5.