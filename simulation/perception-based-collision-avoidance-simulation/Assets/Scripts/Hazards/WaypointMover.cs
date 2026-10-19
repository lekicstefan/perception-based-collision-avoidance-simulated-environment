using System;
using System.Collections.Generic;
using UnityEngine;

// A scripted moving object (pedestrian, crossing or lead vehicle). It follows the waypoints (the children of 'path', in Hierarchy order) in plan view
// and keeps its height. At the start it is placed on the first waypoint.
//
// Motion: it starts at 'speed' and changes it by the 'events' list (ObjectMotion): at a time, at a distance, or after the previous event, instantly (a sudden
// change) or at a rate. A pedestrian that stops half way and then runs is two events.
//
// Start: either 'startDelay' seconds after the run begins, or a time-to-reach trigger. With the trigger the start time is computed before the run so
// that the object reaches the conflict point 'arrivalOffset' seconds after the front of the ego car does (negative: before it). The conflict point is
// a marker object in the scene ('conflictPoint') where the object's path meets the ego route: the distance along both is found by projecting the
// marker on them. The ego car is assumed to drive its scripted speed profile (EgoPrediction), so the start time is the same for every system under
// test and a sweep over the offset gives comparable situations at any scripted speed. The predicted and the achieved arrival times are logged and go
// to run.json (end.outcome.objects), so a dry run shows how well the schedule held.
[DefaultExecutionOrder(-1)]   // before CollisionDetector and RouteDriver (0): the detector must see the objects where they are in this step
public class WaypointMover : MonoBehaviour
{
    public static readonly List<WaypointMover> All = new List<WaypointMover>();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics() { All.Clear(); }

    public Transform path;
    public float speed = 1.4f;          // m/s along the path when it starts
    public float startDelay = 0f;       // s of simulation time before the object starts to move (ignored with a trigger)
    public bool faceDirection = true;   // turn to face the direction of travel
    public bool loop = false;

    [Header("Start trigger (time-to-reach)")]
    public bool useTrigger = false;
    [Tooltip("Seconds by which this object reaches the conflict point after the front of the ego car does. 0 = at the same moment, negative = before it.")]
    public float arrivalOffset = 0f;
    [Tooltip("A marker object where this object's path meets the ego route. Not needed when both distances below are set.")]
    public Transform conflictPoint;
    [Tooltip("Distance along the ego route (m) of the conflict point. -1 = taken from the marker.")]
    public float egoRouteM = -1f;
    [Tooltip("Distance along this object's path (m) of the conflict point. -1 = taken from the marker.")]
    public float objectPathM = -1f;

    [Header("Speed changes")]
    public List<MotionEvent> events = new List<MotionEvent>();

    public bool Started { get; private set; }
    public double StartTime { get; private set; }

    readonly List<Vector3> pts = new List<Vector3>();
    readonly List<Dictionary<string, object>> fired = new List<Dictionary<string, object>>();
    ObjectMotion motion;
    RouteDriver driver;
    Vehicle vehicle;
    bool triggerResolved;
    double startAt;
    float egoFront;                       // m from the rear axle to the front bumper
    float resolvedEgoS = -1f, resolvedObjS = -1f;    // conflict point: route distance of the ego FRONT, path distance of this object
    float predictedEgoArrival, predictedObjTravel;
    double egoArrivedAt = -1.0, objArrivedAt = -1.0;
    bool arrivalLogged;

    void OnEnable() { All.Add(this); }
    void OnDisable() { All.Remove(this); }

    void Start()
    {
        if (path == null) { UnityEngine.Debug.LogWarning("WaypointMover '" + name + "' has no path"); enabled = false; return; }
        foreach (Transform c in path) pts.Add(new Vector3(c.position.x, 0f, c.position.z));
        if (pts.Count < 2)
        {
            UnityEngine.Debug.LogWarning("WaypointMover '" + name + "' needs at least two waypoints");
            enabled = false;
            return;
        }
        Vector3 p = transform.position;
        transform.position = new Vector3(pts[0].x, p.y, pts[0].z);

        motion = new ObjectMotion(pts, speed, events, loop);
        motion.EventFired = OnEventFired;
        Vector3 pos0, dir0;
        motion.Sample(0f, out pos0, out dir0);
        Face(dir0);

        driver = FindFirstObjectByType<RouteDriver>();
        vehicle = FindFirstObjectByType<Vehicle>();
        startAt = startDelay;
    }

    // Resolved once, in the first Update: every Start has run (the route exists) and the handshake with the processor has not finished yet (the
    // simulation is still frozen), so a scenario whose trigger cannot be met stops here, before a run folder exists.
    void Update()
    {
        if (!useTrigger || triggerResolved || motion == null) return;
        triggerResolved = true;
        try { ResolveTrigger(); }
        catch (Exception e)
        {
            enabled = false;
            ScenarioSettings.Fail(new InvalidOperationException("Moving object '" + name + "': " + e.Message, e));
        }
    }

    void ResolveTrigger()
    {
        if (driver == null || driver.Track == null || vehicle == null)
            throw new InvalidOperationException("a time-to-reach trigger needs the RouteDriver (with its route) and the Vehicle in the scene");

        egoFront = FrontOffsetM();
        float sEgo = egoRouteM, sObj = objectPathM;
        if (sEgo < 0f || sObj < 0f)
        {
            if (conflictPoint == null)
                throw new InvalidOperationException("it has a trigger, but no Conflict Point marker and not both egoRouteM and objectPathM");
            Vector3 cp = conflictPoint.position;
            if (sEgo < 0f)
            {
                float d;
                sEgo = ClosestRouteS(driver.Track, cp, out d);
                if (d > 2f) UnityEngine.Debug.LogWarning("TRIGGER " + name + " | the conflict marker is " + d.ToString("F1") + " m from the ego route");
            }
            if (sObj < 0f)
            {
                float d;
                sObj = motion.Project(cp, out d);
                if (d > 2f) UnityEngine.Debug.LogWarning("TRIGGER " + name + " | the conflict marker is " + d.ToString("F1") + " m from this object's path");
            }
        }

        predictedEgoArrival = EgoPrediction.TimeToReach(driver, vehicle, sEgo - egoFront);
        predictedObjTravel = motion.TimeToTravel(sObj, Time.fixedDeltaTime, 600f);
        if (predictedObjTravel < 0f)
            throw new InvalidOperationException("it never reaches path distance " + sObj.ToString("F1") + " m (path length " + motion.PathLength.ToString("F1") + " m)");

        double start = predictedEgoArrival + arrivalOffset - predictedObjTravel;
        if (start < -0.005)
            throw new InvalidOperationException("with arrivalOffset " + arrivalOffset.ToString("0.##") + " s it would have to start at t = " + start.ToString("F2") +
                " s, before the run begins. The ego front reaches the conflict point at t = " + predictedEgoArrival.ToString("F2") + " s and the object needs " +
                predictedObjTravel.ToString("F2") + " s from its start. Move its start further back, or lengthen the approach, or use a larger offset");

        startAt = System.Math.Max(0.0, start);
        resolvedEgoS = sEgo;
        resolvedObjS = sObj;
        UnityEngine.Debug.Log("TRIGGER " + name + " | ego front reaches route " + sEgo.ToString("F1") + " m at t = " + predictedEgoArrival.ToString("F2") +
                              " s (front " + egoFront.ToString("F2") + " m ahead of the rear axle) | object reaches path " + sObj.ToString("F1") + " m " +
                              predictedObjTravel.ToString("F2") + " s after it starts | offset " + arrivalOffset.ToString("+0.00;-0.00;0.00") + " s -> starts at t = " + startAt.ToString("F2") + " s");
    }

    // Distance from the rear axle to the front of the ego car's collision box, along the car.
    float FrontOffsetM()
    {
        CollisionDetector cd = FindFirstObjectByType<CollisionDetector>();
        BoxCollider b = cd != null ? (cd.box != null ? cd.box : cd.GetComponent<BoxCollider>()) : null;
        if (b == null)
        {
            UnityEngine.Debug.LogWarning("TRIGGER " + name + " | no collision box found on the ego car, the front bumper is assumed to be 3.5 m ahead of the rear axle");
            return 3.5f;
        }
        Vector3 frontLocal = b.center + new Vector3(0f, 0f, b.size.z * 0.5f);
        float z = vehicle.transform.InverseTransformPoint(b.transform.TransformPoint(frontLocal)).z;
        if (z < 0.5f)
        {
            UnityEngine.Debug.LogWarning("TRIGGER " + name + " | the front of the collision box is " + z.ToString("F2") + " m ahead of the rear axle, which looks wrong (is the box's local z the car's length?)");
            return Mathf.Max(z, 0f);
        }
        return z;
    }

    // Route distance of the point on the route closest to p (plan view), searched over the whole route.
    static float ClosestRouteS(Route route, Vector3 p, out float distance)
    {
        float bestD = float.MaxValue, bestS = 0f;
        for (float s = 0f; s <= route.Length; s += 0.5f)
        {
            Vector3 q = route.Sample(s).position;
            float d = (q.x - p.x) * (q.x - p.x) + (q.z - p.z) * (q.z - p.z);
            if (d < bestD) { bestD = d; bestS = s; }
        }
        bestS = route.FindClosestS(p.x, p.z, bestS, 1f, 1f);
        Vector3 r = route.Sample(bestS).position;
        distance = Mathf.Sqrt((r.x - p.x) * (r.x - p.x) + (r.z - p.z) * (r.z - p.z));
        return bestS;
    }

    void OnEventFired(int index, double motionTime, float newSpeed)
    {
        double now = SimClock.Now;
        fired.Add(new Dictionary<string, object> { { "index", index }, { "t", now }, { "speedMps", newSpeed } });
        UnityEngine.Debug.Log("MOTION " + name + " | event " + index + " at t = " + now.ToString("F2") + " s (" + motionTime.ToString("F2") + " s after its start, " +
                              motion.Travelled.ToString("F1") + " m travelled): target speed " + newSpeed.ToString("0.##") + " m/s, " +
                              (events[index].accel > 0f ? "accel " + events[index].accel.ToString("0.##") + " m/s^2" : "at once"));
    }

    void Face(Vector3 dir)
    {
        dir.y = 0f;
        if (faceDirection && dir.sqrMagnitude > 1e-8f)
            transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
    }

    void FixedUpdate()
    {
        if (motion == null) return;
        double now = SimClock.Now;

        // how well the schedule holds: when the front of the ego car and this object really reach the conflict point
        if (triggerResolved && resolvedEgoS >= 0f)
        {
            if (egoArrivedAt < 0.0 && driver != null && driver.S + egoFront >= resolvedEgoS) egoArrivedAt = now;
            if (!arrivalLogged && egoArrivedAt >= 0.0 && objArrivedAt >= 0.0)
            {
                arrivalLogged = true;
                UnityEngine.Debug.Log("TRIGGER " + name + " | achieved: ego front at the conflict point at t = " + egoArrivedAt.ToString("F2") + " s, object at t = " +
                                      objArrivedAt.ToString("F2") + " s | offset " + (objArrivedAt - egoArrivedAt).ToString("+0.00;-0.00;0.00") + " s (asked " + arrivalOffset.ToString("+0.00;-0.00;0.00") + " s)");
            }
        }

        if (!Started)
        {
            if (useTrigger && !triggerResolved) return;
            if (now < startAt) return;
            Started = true;
            StartTime = now;
        }

        float dt = Time.fixedDeltaTime;
        motion.Step(dt);

        Vector3 pos, dir;
        motion.Sample(motion.S, out pos, out dir);
        transform.position = new Vector3(pos.x, transform.position.y, pos.z);
        Face(dir);

        if (triggerResolved && resolvedObjS >= 0f && objArrivedAt < 0.0 && motion.Travelled >= resolvedObjS) objArrivedAt = now;
    }

    // What this object did in the run, for run.json.
    public Dictionary<string, object> Report()
    {
        var d = new Dictionary<string, object>
        {
            { "startTimeS", Started ? (object)StartTime : null },
            { "pathTravelledM", motion != null ? motion.Travelled : 0f },
        };
        if (useTrigger && resolvedEgoS >= 0f)
        {
            var t = new Dictionary<string, object>
            {
                { "arrivalOffsetS", arrivalOffset },
                { "egoRouteM", resolvedEgoS },
                { "objectPathM", resolvedObjS },
                { "egoFrontOffsetM", egoFront },
                { "predictedEgoArrivalS", predictedEgoArrival },
                { "predictedObjectTravelS", predictedObjTravel },
                { "predictedStartS", startAt },
                { "achievedEgoArrivalS", egoArrivedAt >= 0.0 ? (object)egoArrivedAt : null },
                { "achievedObjectArrivalS", objArrivedAt >= 0.0 ? (object)objArrivedAt : null },
                { "achievedOffsetS", egoArrivedAt >= 0.0 && objArrivedAt >= 0.0 ? (object)(objArrivedAt - egoArrivedAt) : null },
            };
            d["trigger"] = t;
        }
        if (fired.Count > 0) d["events"] = fired;
        return d;
    }

    public static SortedDictionary<string, object> CollectReports()
    {
        var all = new SortedDictionary<string, object>(StringComparer.Ordinal);
        foreach (WaypointMover m in All) all[m.name] = m.Report();
        return all;
    }

    void OnDrawGizmos()
    {
        if (path == null) return;
        Gizmos.color = Color.magenta;
        Transform prev = null;
        foreach (Transform c in path)
        {
            Gizmos.DrawSphere(c.position, 0.2f);
            if (prev != null) Gizmos.DrawLine(prev.position, c.position);
            prev = c;
        }
        if (useTrigger && conflictPoint != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(conflictPoint.position, 0.6f);
            Gizmos.DrawLine(conflictPoint.position, conflictPoint.position + Vector3.up * 3f);
        }
    }
}