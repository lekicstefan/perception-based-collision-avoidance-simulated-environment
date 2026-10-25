using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

[DefaultExecutionOrder(-300)]   // before SensorRig (-200) and Vehicle (-100), whose Awake read the values set here
public class ScenarioSettings : MonoBehaviour
{
    [Tooltip("Scenario file in configs/scenarios, without .json (or a path). --scenario overrides it. Empty: the scene runs exactly as built and no scenario is recorded.")]
    public string scenario = "dev";

    [Tooltip("Editor convenience, one per line, for example driver.cruiseSpeedMps=10 or objects.CrossingVehicle.startDelayS=2. Applied before the command line ones.")]
    public string[] editorOverrides = new string[0];

    [Header("Wiring (found in the scene when left empty)")]
    public Vehicle vehicle;
    public RouteDriver driver;
    public CommandExecutor executor;
    public SensorRig rig;
    public EnvironmentSettings environment;   // optional

    ScenarioConfig cfg;
    readonly SortedDictionary<string, WaypointMover> movers = new SortedDictionary<string, WaypointMover>(StringComparer.Ordinal);

    void Awake()
    {
        string name = !string.IsNullOrEmpty(ScenarioContext.Pick) ? ScenarioContext.Pick : Cli.Get("--scenario", scenario);   // the menu pick wins, then --scenario, then the field
        if (string.IsNullOrEmpty(name))
        {
            UnityEngine.Debug.LogWarning("SCENARIO: none selected, the scene runs exactly as built and no scenario is recorded");
            return;
        }
        try
        {
            FindComponents();

            var overrides = new List<string>();
            foreach (string o in editorOverrides) if (!string.IsNullOrWhiteSpace(o)) overrides.Add(o.Trim());
            overrides.AddRange(Cli.GetAll("--set"));

            string path;
            cfg = ScenarioLoader.Load(name, overrides, out path);

            string sceneName = SceneManager.GetActiveScene().name;
            if (cfg.scene != sceneName)
                throw new ArgumentException("Scenario '" + cfg.name + "' belongs to the scene '" + cfg.scene + "' but the scene '" + sceneName + "' is loaded");

            Apply(cfg);
            ScenarioContext.SetLoaded(cfg, path, overrides.ToArray());
            UnityEngine.Debug.Log("SCENARIO '" + cfg.name + "' (" + path + ") | scene " + sceneName + " | " + cfg.objects.Count + " object(s) configured | overrides: " +
                                  (overrides.Count == 0 ? "none" : string.Join("; ", overrides)));
        }
        catch (Exception e)
        {
            cfg = null;
            Fail(e);
        }
    }

    void Start()
    {
        if (cfg == null) return;
        try { ScenarioContext.SetResolved(ScenarioLoader.ToJson(BuildResolved())); }
        catch (Exception e) { Fail(e); }
    }

    void FindComponents()
    {
        if (vehicle == null) vehicle = FindFirstObjectByType<Vehicle>();
        if (driver == null) driver = FindFirstObjectByType<RouteDriver>();
        if (executor == null) executor = FindFirstObjectByType<CommandExecutor>();
        if (rig == null) rig = FindFirstObjectByType<SensorRig>();
        if (environment == null) environment = FindFirstObjectByType<EnvironmentSettings>();
        if (vehicle == null) throw new InvalidOperationException("ScenarioSettings: no Vehicle in the scene");
        if (driver == null) throw new InvalidOperationException("ScenarioSettings: no RouteDriver in the scene");
        if (executor == null) throw new InvalidOperationException("ScenarioSettings: no CommandExecutor in the scene");
        if (rig == null) throw new InvalidOperationException("ScenarioSettings: no SensorRig in the scene");

        movers.Clear();
        foreach (WaypointMover m in FindObjectsByType<WaypointMover>(FindObjectsSortMode.None))
        {
            if (movers.ContainsKey(m.gameObject.name))
                throw new InvalidOperationException("ScenarioSettings: two moving objects are called '" + m.gameObject.name + "', give them different names");
            movers.Add(m.gameObject.name, m);
        }
    }

    static void Set(float? v, ref float target) { if (v.HasValue) target = v.Value; }
    static void Set(bool? v, ref bool target) { if (v.HasValue) target = v.Value; }

    void Apply(ScenarioConfig c)
    {
        DriverParams d = c.driver;
        if (d != null)
        {
            if (!string.IsNullOrEmpty(d.route)) driver.routeFile = d.route;
            Set(d.cruiseSpeedMps, ref driver.cruiseSpeed);
            Set(d.initialSpeedMps, ref driver.initialSpeed);
            Set(d.maxLateralAccelMps2, ref driver.maxLateralAccel);
            Set(d.comfortDecelMps2, ref driver.comfortDecel);
            Set(d.comfortAccelMps2, ref driver.comfortAccel);
            Set(d.speedGain, ref driver.speedGain);
            Set(d.driverMaxBrakeMps2, ref driver.driverMaxBrake);
            Set(d.lookaheadMinM, ref driver.lookaheadMin);
            Set(d.lookaheadPerSpeedS, ref driver.lookaheadPerSpeed);
            Set(d.lookaheadMaxM, ref driver.lookaheadMax);
            Set(d.endMarginM, ref driver.endMargin);
        }

        VehicleParams v = c.vehicle;
        if (v != null)
        {
            Set(v.wheelbaseM, ref vehicle.wheelbase);
            Set(v.maxSteeringDeg, ref vehicle.maxSteeringDeg);
            Set(v.maxSteeringRateDegPerS, ref vehicle.maxSteeringRateDeg);
            Set(v.maxAccelMps2, ref vehicle.maxAccel);
            Set(v.maxBrakeMps2, ref vehicle.maxBrake);
            Set(v.actuationDelayS, ref vehicle.actuationDelay);
            Set(v.jerkLimitMps3, ref vehicle.jerkLimit);
        }

        SensorParams s = c.sensors;
        if (s != null)
        {
            if (!string.IsNullOrEmpty(s.config)) rig.configName = s.config;
            if (s.seed.HasValue) rig.seedOverride = s.seed.Value;
        }

        EnvironmentParams env = c.environment;
        if (env != null && env.fogVisibilityM.HasValue)
        {
            if (environment != null) environment.fogVisibilityM = env.fogVisibilityM.Value;   // its Awake copies this
            SimEnvironment.FogVisibilityM = env.fogVisibilityM.Value;
        }

        if (c.objects != null)
            foreach (KeyValuePair<string, ObjectParams> kv in c.objects)
            {
                WaypointMover m;
                if (!movers.TryGetValue(kv.Key, out m))
                    throw new ArgumentException("Scenario '" + c.name + "' lists the object '" + kv.Key + "' but no moving object (WaypointMover) of that name is in the scene. Moving objects here: " +
                                                (movers.Count > 0 ? string.Join(", ", new List<string>(movers.Keys)) : "(none)"));
                ObjectParams o = kv.Value;
                if (o == null) continue;
                Set(o.speedMps, ref m.speed);
                Set(o.startDelayS, ref m.startDelay);
                Set(o.loop, ref m.loop);
                Set(o.faceDirection, ref m.faceDirection);
                if (o.trigger != null)
                {
                    m.useTrigger = true;
                    Set(o.trigger.arrivalOffsetS, ref m.arrivalOffset);
                    Set(o.trigger.egoRouteM, ref m.egoRouteM);
                    Set(o.trigger.objectPathM, ref m.objectPathM);
                }
                if (o.events != null) m.events = ToEvents(o.events);
            }
    }

    static List<MotionEvent> ToEvents(List<MotionEventParams> src)
    {
        var list = new List<MotionEvent>();
        foreach (MotionEventParams p in src)
        {
            var e = new MotionEvent { speed = p.speedMps.Value, accel = p.accelMps2 ?? 0f };
            if (p.atTimeS.HasValue) { e.timing = MotionEvent.Timing.AtTime; e.value = p.atTimeS.Value; }
            else if (p.atPathM.HasValue) { e.timing = MotionEvent.Timing.AtPath; e.value = p.atPathM.Value; }
            else { e.timing = MotionEvent.Timing.AfterPrevious; e.value = p.afterPreviousS.Value; }
            list.Add(e);
        }
        return list;
    }

    static List<MotionEventParams> FromEvents(List<MotionEvent> src)
    {
        var list = new List<MotionEventParams>();
        foreach (MotionEvent e in src)
        {
            var p = new MotionEventParams { speedMps = e.speed, accelMps2 = e.accel };
            if (e.timing == MotionEvent.Timing.AtTime) p.atTimeS = e.value;
            else if (e.timing == MotionEvent.Timing.AtPath) p.atPathM = e.value;
            else p.afterPreviousS = e.value;
            list.Add(p);
        }
        return list;
    }

    ScenarioConfig BuildResolved()
    {
        var r = new ScenarioConfig
        {
            schema = cfg.schema,
            name = cfg.name,
            description = cfg.description,
            scene = cfg.scene,
            systemVariant = cfg.systemVariant,
            end = cfg.end
        };

        r.driver.route = driver.routeFile;
        r.driver.cruiseSpeedMps = driver.cruiseSpeed;
        r.driver.initialSpeedMps = driver.initialSpeed;
        r.driver.maxLateralAccelMps2 = driver.maxLateralAccel;
        r.driver.comfortDecelMps2 = driver.comfortDecel;
        r.driver.comfortAccelMps2 = driver.comfortAccel;
        r.driver.speedGain = driver.speedGain;
        r.driver.driverMaxBrakeMps2 = driver.driverMaxBrake;
        r.driver.lookaheadMinM = driver.lookaheadMin;
        r.driver.lookaheadPerSpeedS = driver.lookaheadPerSpeed;
        r.driver.lookaheadMaxM = driver.lookaheadMax;
        r.driver.endMarginM = driver.endMargin;

        r.vehicle.wheelbaseM = vehicle.wheelbase;
        r.vehicle.maxSteeringDeg = vehicle.maxSteeringDeg;
        r.vehicle.maxSteeringRateDegPerS = vehicle.maxSteeringRateDeg;
        r.vehicle.maxAccelMps2 = vehicle.maxAccel;
        r.vehicle.maxBrakeMps2 = vehicle.maxBrake;
        r.vehicle.actuationDelayS = vehicle.actuationDelay;
        r.vehicle.jerkLimitMps3 = vehicle.jerkLimit;

        r.sensors.config = rig.Config != null ? rig.Config.name : rig.configName;   // final: the command line may have replaced it
        r.sensors.seed = rig.Seed;

        r.environment.fogVisibilityM = SimEnvironment.FogVisibilityM;

        foreach (KeyValuePair<string, WaypointMover> kv in movers)
        {
            WaypointMover m = kv.Value;
            var o = new ObjectParams
            {
                speedMps = m.speed,
                loop = m.loop,
                faceDirection = m.faceDirection,
                events = m.events != null && m.events.Count > 0 ? FromEvents(m.events) : null
            };
            if (m.useTrigger)       // with a trigger the start time is computed (see run.json, end.outcome.objects), startDelayS does not apply
                o.trigger = new TriggerParams
                {
                    arrivalOffsetS = m.arrivalOffset,
                    egoRouteM = m.egoRouteM >= 0f ? (float?)m.egoRouteM : null,
                    objectPathM = m.objectPathM >= 0f ? (float?)m.objectPathM : null
                };
            else o.startDelayS = m.startDelay;
            r.objects[kv.Key] = o;
        }
        return r;
    }

    public static void Fail(Exception e)
    {
        UnityEngine.Debug.LogError("SCENARIO ERROR: " + e.Message + "\nThe run is stopped.");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        UnityEngine.Application.Quit(1);
#endif
    }
}