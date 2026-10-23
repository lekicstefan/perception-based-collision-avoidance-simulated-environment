# Scenarios

A scenario is **one Unity scene plus one JSON file**. The world (road, parked cars, people, lamp posts, buildings) is built in the editor, one scene per scenario. A scenario scene starts as a copy of `DevTestScene` (ego car, sensors, scenario manager and views come along): the route file gives the path, the road is built along it, the roadside scenery is generated along the road (docs/scenery.md), and the objects that belong to this scenario are placed by hand. The JSON file in `configs/scenarios/<name>.json` holds the parameters: speeds, timing, fog, sensor configuration, end conditions. Scenario files are committed; run folders are not.

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
| `driver` | `route`, `cruiseSpeedMps`, `initialSpeedMps` (speed at the start of the route, 0 or missing = from rest), `maxLateralAccelMps2`, `comfortDecelMps2`, `comfortAccelMps2`, `speedGain`, `driverMaxBrakeMps2`, `lookaheadMinM`, `lookaheadPerSpeedS`, `lookaheadMaxM`, `endMarginM` (the scripted driver, `RouteDriver`) |
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

Limits: the prediction ignores steering (the route arc length is used as the distance driven), so it is a little off on tight bends; the conflict point must be before the point where the driver stops (route end minus `endMarginM`); the ego car starts at rest or, with `driver.initialSpeedMps`, at that speed (the speed profile then starts from it, so it keeps its speed instead of braking).

## Scenarios

Every scenario scene is a copy of another scenario scene (S1 for the straight road, S4 for a crossing, S5 for the narrow street) with its objects placed by hand. The ego car drives the scripted route at the given speed; the scripted driver does not react to anything, so a collision in a run without the system is the expected outcome. All scenarios use the sensor configuration `default`, seed 12345, a 60 s timeout, and start the ego car at its cruise speed (`driver.initialSpeedMps`). Pedestrian, cat and vehicle starts use the time-to-reach trigger with `arrivalOffsetS` 0 unless stated.

| Scene | Scenario file | Route | Ego speed | What happens |
|---|---|---|---|---|
| `S1_StationaryObstacle` | `s1_stationary_obstacle` | `scn_straight_300` | 50 km/h | a parked car (`StationaryCar`) stands in the ego lane 200 m from the start |
| | `s1_stationary_obstacle_fog100` | | 50 km/h | the same with 100 m fog visibility |
| `S2_SlowLeadVehicle` | `s2_slow_lead_vehicle` | `scn_straight_300` | 50 km/h | `LeadCar` 30 m ahead drives 20 km/h and never brakes |
| `S3_BrakingLeadVehicle` | `s3_braking_lead_vehicle` | `scn_straight_300` | 50 km/h | `LeadCar` 25 m ahead drives 50 km/h and brakes at 6 m/s^2 to a stop after 3 s |
| `S4_CrossingPedestrian` | `s4_crossing_pedestrian` | `scn_straight_300` | 50 km/h | an adult walks 1.4 m/s from the right pavement across the lane, collision course; zebra crossing at 100 m |
| | `s4_crossing_pedestrian_fog100` | | 50 km/h | the same with 100 m fog |
| | `s4_sudden_change` | | 50 km/h | the pedestrian stops at the kerb edge (1.5 m along its path) for 1.5 s, then runs at 3.5 m/s (6 m/s^2) |
| `S5_NarrowStreet` | `s5_narrow_street` | `scn_straight_300` | 30 km/h | cars parked along the right kerb, the ego car in the middle of the road; an adult steps out between two parked cars at 1.5 m/s |
| `S6_NarrowStreetStanding` | `s6_narrow_street_standing` | `scn_straight_300` | 30 km/h | a person stands next to a parked car, touching it; nobody moves |
| `S7_NegativeControl` | `s7_negative_control` | `scn_straight_300` | 30 km/h | no hazard: parked cars, a person on the pavement, an oncoming car at 50 km/h in the other lane |
| `S8_CatCrossing` | `s8_cat_crossing` | `scn_straight_300` | 50 km/h | a cat (0.5 m long) runs 3 m/s from the right kerb across the lane, collision course |
| `S9_CrestHiddenVehicle` | `s9_crest_hidden_vehicle` | `scn_crest` | 50 km/h | the road climbs 8 % to a crest at 150 m (peak 6.9 m); a stopped car stands in the lane at 175 m, hidden by the crest until about 40 to 50 m |
| `S10_BlindBend` | `s10_blind_bend` | `scn_bend` | 40 km/h | straight 100 m, left bend (90 degrees, radius 30 m), a stopped car in the lane after the bend, hidden by the buildings on the inside |

Relation to the specification: scenarios 1 to 4 are the same; specification 5 is `S5` plus its standing-person variant `S6`; specification 6 (fog) is the `_fog100` files, other densities come from `--set environment.fogVisibilityM=...`; specification 7 (sudden change, pedestrian version) is `s4_sudden_change`; specification 8 (negative controls) is `S7`. `S8` (cat) is an addition (an object much smaller than a person). `S9` and `S10` are the extended scenarios 9 (crest) and 10 (blind bend).

Routes (`configs/routes`): `scn_straight_300` is 300 m straight. `scn_crest` is straight with the elevation profile above (the waypoints are 10 m apart, the route is splined). `scn_bend` has a bend of 30 m radius; the scripted driver slows for it by its lateral acceleration limit. The road is 9 m wide with a centre offset of 2.25 m (the ego car in the right lane) in S1 to S4 and S8 to S10, and with offset 0 (the ego car in the middle) in S5 to S7.

Sweep points of a scenario are made with `--set` (see below): ego speed (`driver.cruiseSpeedMps` and `driver.initialSpeedMps`), the trigger offset, object speeds, event times, fog.

## Overrides and sweep points

    --set driver.cruiseSpeedMps=11.1
    --set objects.CrossingVehicle.trigger.arrivalOffsetS=-0.5
    --set objects.Ped.events.0.atTimeS=2.5
    --set environment.fogVisibilityM=200

The path is the JSON path with dots. The value is read as JSON (number, `true`, `false`) or else as text. This is how the batch runner (phase 14) creates a sweep point from one scenario file. The overrides are listed in `run.json`.

## Running a scenario: run lifecycle and controls

Every scenario scene has a `ScenarioManager`. A run starts when the processor answers READY. It ends at the first of: collision (the car is frozen at the impact), route finished (the driver reached the end and the car stands still), stopped (the car drove and then stood still for `end.stoppedForS`), timeout (`end.timeoutS` of simulation time), or manual (restart, End run). The manager then tells the processor (END_OF_RUN), writes the outcome to `run.json` (`end.outcome`: reason, sim time, final speed, route progress, impact speed and hazard, the report of every moving object, and whether the run is valid for experiments), waits until the logs are closed and freezes the simulation.

Run the processor with `--loop` so that it waits for the next run: `python -m server.run_processor --pin 2,4 --run-dir name --loop`. Each run gets its own folder (`name`, `name_2`, ...). Unity reloads the scene on restart (the scene must be in the Build Settings). `--quit-on-end` (Unity command line) quits the player when the run has ended, for batch runs.

Keys (click the Game view first): `Esc` back to the menu, `Space` pause / resume, `E` end the run, `R` restart, `Up` / `Right` and `Down` / `Left` speed 0.25x to 4x, `0` back to 1x. A run at any speed other than 1x is **not valid for experiments** (both programs run on real wall-clock time) and `run.json` says so; every run starts at 1x. Pauses are recorded (`pauses`, `pausedRealS`) and do not invalidate a run.

### Menu

The scene `Menu` (`Assets/Scenes/Menu.unity`, first in the Build Settings) has one object with `ScenarioMenu` (`Assets/Scripts/Scenario/`). It lists every file in `configs/scenarios` with its scene and description. **Start** loads the scene of that scenario and makes it the scenario that the scene runs: the choice (`ScenarioContext.Pick`) wins over `--scenario` and over the `Scenario` field of the scene, until the play session ends, so a restart (`R`) repeats the same scenario. `Esc` or the **Menu** button in a scenario scene ends the run and comes back to the list. A scenario whose scene is not in the Build Settings is greyed out with the reason; a file with an error shows the error. Scenes can still be opened and played directly, as before.

The menu does not start the processor. Start it first with `--loop`; a scenario scene waits for it, and every scene load makes a new handshake and a new run folder.

Views (`ViewControls`, debug only, never sent to the processor): `1`-`9` or `W` / `Shift+W` camera, `Q` camera view, `A` lidar view (off / range image / bird's-eye / both), `S` lidar dots, `D` lidar beams (row / grid / objects), `F` `G` move the beam row, `Z` dashboard, `X` minimap, `C` tracks as boxes on the camera picture, `V` route, `B` hazard boxes, `Tab` hide all panels and pictures, `Shift+Tab` hide the effects in the world (they stay switched on). Dots, dashboard, minimap and track overlay start off. The choices are kept across restarts.

## Notes

- The processor assumes the braking capability measured in phase 2 (8 m/s^2, 0.15 s effective delay). `vehicle.maxBrakeMps2` and `vehicle.actuationDelayS` are in the file so that a run documents the car it used; changing them changes the car, not the processor's model.
- The scenario does not set the processor's options (`--passing-speed-kmh`, `--pin`, pose noise sigmas). The batch runner passes those.
- `dev` is the reference file for the development scene, `dev_trigger` tests the triggers and speed events there. The core scenarios are listed below (see `docs/scenery.md` for the scenery tools).