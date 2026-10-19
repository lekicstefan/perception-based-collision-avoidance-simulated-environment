using System.Collections.Generic;
using UnityEngine;

// One speed change of a moving object: from the moment it fires, the object heads for a new speed.
//   timing AtTime          fires when the object has been moving for 'value' seconds (counted from the start of its motion)
//   timing AtPath          fires when the object has travelled 'value' metres along its path
//   timing AfterPrevious   fires 'value' seconds after the previous event fired (after the start, for the first event)
// accel is how fast the speed changes (m/s^2); 0 or less means at once, a sudden change.
// Events fire in list order, each one only after the one before it.
[System.Serializable]
public class MotionEvent
{
    public enum Timing { AtTime, AtPath, AfterPrevious }

    public Timing timing = Timing.AtTime;
    public float value;
    public float speed;      // m/s, the new target speed
    public float accel;      // m/s^2, 0 = at once
}

// The motion of a scripted object along a polyline (plan view): speed, speed events and position. No Unity scene objects are used, so the
// same code runs in the scene (WaypointMover) and as a dry run that predicts when the object reaches a point (time-to-reach triggers).
public class ObjectMotion
{
    readonly Vector3[] pts;
    readonly float[] cum;                 // cum[i] = distance along the path of point i
    readonly MotionEvent[] events;
    readonly float cruise;
    readonly bool loop;

    float accel;
    double lastEventTime;

    public float PathLength { get; private set; }
    public float S { get; private set; }            // position along the path (wraps to 0 when the path loops)
    public float Travelled { get; private set; }    // distance since the start of the motion (does not wrap)
    public float Speed { get; private set; }
    public float TargetSpeed { get; private set; }
    public double Elapsed { get; private set; }     // seconds since the start of the motion
    public int NextEvent { get; private set; }
    public bool AtEnd { get; private set; }         // reached the end of a path that does not loop

    // index of the event, time since the start of the motion, new target speed
    public System.Action<int, double, float> EventFired;

    public ObjectMotion(IList<Vector3> points, float cruiseSpeed, IList<MotionEvent> motionEvents, bool loops)
    {
        pts = new Vector3[points.Count];
        for (int i = 0; i < pts.Length; i++) pts[i] = new Vector3(points[i].x, 0f, points[i].z);
        cum = new float[pts.Length];
        for (int i = 1; i < pts.Length; i++) cum[i] = cum[i - 1] + (pts[i] - pts[i - 1]).magnitude;
        PathLength = pts.Length > 0 ? cum[pts.Length - 1] : 0f;

        events = new MotionEvent[motionEvents != null ? motionEvents.Count : 0];
        for (int i = 0; i < events.Length; i++) events[i] = motionEvents[i];
        cruise = cruiseSpeed;
        loop = loops;
        Speed = TargetSpeed = cruiseSpeed;
    }

    // One physics step. The speed moves toward the target, the distance of the step is the average speed times dt.
    public void Step(float dt)
    {
        Elapsed += dt;

        while (NextEvent < events.Length)
        {
            MotionEvent e = events[NextEvent];
            bool due;
            if (e.timing == MotionEvent.Timing.AtTime) due = Elapsed >= e.value;
            else if (e.timing == MotionEvent.Timing.AtPath) due = Travelled >= e.value;
            else due = Elapsed >= lastEventTime + e.value;
            if (!due) break;

            TargetSpeed = e.speed;
            accel = e.accel;
            lastEventTime = Elapsed;
            int index = NextEvent++;
            if (EventFired != null) EventFired(index, Elapsed, e.speed);
        }

        float vOld = Speed;
        if (accel <= 0f) Speed = TargetSpeed;
        else if (Speed < TargetSpeed) Speed = Mathf.Min(TargetSpeed, Speed + accel * dt);
        else Speed = Mathf.Max(TargetSpeed, Speed - accel * dt);

        float ds = 0.5f * (vOld + Speed) * dt;
        if (!loop && S + ds >= PathLength)
        {
            ds = Mathf.Max(0f, PathLength - S);
            AtEnd = true;
            Speed = 0f;
        }
        S += ds;
        Travelled += ds;
        if (loop && PathLength > 0f && S >= PathLength) S -= PathLength;
    }

    // Position and direction of travel at a distance along the path.
    public void Sample(float s, out Vector3 position, out Vector3 direction)
    {
        s = Mathf.Clamp(s, 0f, PathLength);
        int i = 0;
        while (i < pts.Length - 2 && cum[i + 1] < s) i++;
        float len = cum[i + 1] - cum[i];
        float a = len > 1e-6f ? (s - cum[i]) / len : 0f;
        position = Vector3.Lerp(pts[i], pts[i + 1], a);
        direction = pts[i + 1] - pts[i];
        direction.y = 0f;
        if (direction.sqrMagnitude > 1e-10f) direction.Normalize();
    }

    // Distance along the path of the point closest to p (plan view), and how far p is from the path.
    public float Project(Vector3 p, out float distance)
    {
        float bestD = float.MaxValue, bestS = 0f;
        for (int i = 0; i + 1 < pts.Length; i++)
        {
            Vector3 a = pts[i], ab = pts[i + 1] - pts[i];
            float len2 = ab.x * ab.x + ab.z * ab.z;
            float t = len2 > 1e-10f ? Mathf.Clamp01(((p.x - a.x) * ab.x + (p.z - a.z) * ab.z) / len2) : 0f;
            float qx = a.x + t * ab.x - p.x, qz = a.z + t * ab.z - p.z;
            float d = qx * qx + qz * qz;
            if (d < bestD)
            {
                bestD = d;
                bestS = cum[i] + t * (cum[i + 1] - cum[i]);
            }
        }
        distance = Mathf.Sqrt(bestD);
        return bestS;
    }

    // Dry run: seconds after the start of the motion until the object has travelled sTarget metres (with its speed events), or -1 when it never does.
    public float TimeToTravel(float sTarget, float dt, float maxSeconds)
    {
        if (sTarget <= 0f) return 0f;
        var m = new ObjectMotion(pts, cruise, events, loop);
        while (m.Elapsed < maxSeconds)
        {
            float before = m.Travelled;
            m.Step(dt);
            if (m.Travelled >= sTarget)
            {
                float f = m.Travelled > before ? (sTarget - before) / (m.Travelled - before) : 1f;
                return (float)(m.Elapsed - dt + f * dt);
            }
            if (m.AtEnd) break;
        }
        return -1f;
    }
}