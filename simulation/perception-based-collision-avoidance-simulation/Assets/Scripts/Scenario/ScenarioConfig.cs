using System.Collections.Generic;

// Scenario configuration (format: docs/scenarios.md). One scenario = one Unity scene (the world is built by hand in the
// editor) + one JSON file in configs/scenarios (the parameters). Every parameter is optional: a missing one leaves the
// value that is set on the component in the scene. The fully resolved configuration (every value that was actually
// used) is what a run stores, see ScenarioSettings.
// The JSON keys are exactly the field names below, a misspelled key is an error (ScenarioLoader).
public class ScenarioConfig
{
    public int schema = 1;
    public string name;                       // must equal the file name without .json
    public string description = "";
    public string scene;                      // name of the Unity scene this scenario belongs to (checked at start)
    public string systemVariant = "default";  // which processor variant runs (phase 9); only recorded for now

    public DriverParams driver = new DriverParams();
    public VehicleParams vehicle = new VehicleParams();
    public ExecutorParams executor = new ExecutorParams();
    public SensorParams sensors = new SensorParams();
    public EnvironmentParams environment = new EnvironmentParams();
    public Dictionary<string, ObjectParams> objects = new Dictionary<string, ObjectParams>();   // key = name of the scene object
    public EndParams end = new EndParams();
}

// The scripted driver (RouteDriver).
public class DriverParams
{
    public string route;                  // route file under configs, for example routes/dev_straight.json
    public float? cruiseSpeedMps;
    public float? initialSpeedMps;        // speed at the start of the route (missing or 0 = starts from rest)
    public float? maxLateralAccelMps2;
    public float? comfortDecelMps2;
    public float? comfortAccelMps2;
    public float? speedGain;              // 1/s
    public float? driverMaxBrakeMps2;
    public float? lookaheadMinM;
    public float? lookaheadPerSpeedS;
    public float? lookaheadMaxM;
    public float? endMarginM;
}

// The vehicle plant (Vehicle). The processor assumes the braking capability of phase 2 (8 m/s^2, 0.15 s effective delay):
// changing maxBrakeMps2 or actuationDelayS changes the car, not the processor.
public class VehicleParams
{
    public float? wheelbaseM;
    public float? maxSteeringDeg;
    public float? maxSteeringRateDegPerS;
    public float? maxAccelMps2;
    public float? maxBrakeMps2;
    public float? actuationDelayS;
    public float? jerkLimitMps3;
}

// The command executor (CommandExecutor).
public class ExecutorParams
{
    public float? capGain;                // 1/s
    public bool? enforceExpiry;           // false = ablation "expiring commands off"
}

// Sensors: which sensor configuration file (configs/sensors) and which base seed. The command line (--sensors, --seed) wins.
public class SensorParams
{
    public string config;
    public int? seed;
}

public class EnvironmentParams
{
    public float? fogVisibilityM;         // meteorological visibility, 0 = clear air
}

// A moving scene object (WaypointMover) found by the name of its GameObject.
public class ObjectParams
{
    public float? speedMps;               // speed when it starts to move
    public float? startDelayS;            // seconds after the run begins; not together with a trigger
    public bool? loop;
    public bool? faceDirection;
    public TriggerParams trigger;         // start so that it reaches the conflict point at a time offset from the ego car
    public List<MotionEventParams> events;   // speed changes; a list in the file replaces the one set in the scene
}

// Time-to-reach start trigger. The conflict point is a marker in the scene (WaypointMover.conflictPoint) unless both distances are given here.
public class TriggerParams
{
    public float? arrivalOffsetS;         // the object reaches the conflict point this many seconds AFTER the front of the ego car (negative: before). Required
    public float? egoRouteM;              // distance along the ego route of the conflict point (replaces the marker for the ego side)
    public float? objectPathM;            // distance along the object's path of the conflict point (replaces the marker for the object side)
}

// One speed change. Exactly one of atTimeS, atPathM, afterPreviousS says when it happens.
public class MotionEventParams
{
    public float? atTimeS;                // seconds since the object started to move
    public float? atPathM;                // metres travelled along its path
    public float? afterPreviousS;         // seconds after the previous event (after the start, for the first)
    public float? speedMps;               // the new target speed. Required
    public float? accelMps2;              // m/s^2 toward the new speed; missing or 0 = at once (a sudden change)
}

// When a run ends. Enforced by the scenario manager (ScenarioManager).
public class EndParams
{
    public float timeoutS = 120f;         // simulation time
    public bool onCollision = true;
    public bool onRouteFinished = true;
    public float stoppedSpeedMps = 0.1f;  // "stopped" = slower than this ...
    public float stoppedForS = 3f;        // ... for this long (0 = this condition is off)
}