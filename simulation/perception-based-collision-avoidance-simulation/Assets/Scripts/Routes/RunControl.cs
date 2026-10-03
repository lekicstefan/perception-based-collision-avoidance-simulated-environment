using UnityEngine;

// Run end state. The scenario manager (phase 8) will add the other end conditions.
public static class RunControl
{
    public enum EndReason { None, Collision, Stopped, RouteFinished, Timeout, Manual }

    public static bool Ended { get; private set; }
    public static EndReason Reason { get; private set; }
    public static double EndTime { get; private set; }
    public static float ImpactSpeed { get; private set; }   // m/s, ego speed at the first overlap
    public static string HazardName { get; private set; }
    public static Vector3 EgoPosition { get; private set; }

    public static void Reset()
    {
        Ended = false;
        Reason = EndReason.None;
        EndTime = 0.0;
        ImpactSpeed = 0f;
        HazardName = null;
        EgoPosition = Vector3.zero;
    }

    public static void End(EndReason reason, double time)
    {
        if (Ended) return;
        Ended = true;
        Reason = reason;
        EndTime = time;
    }

    public static void EndWithCollision(double time, float impactSpeed, string hazardName, Vector3 egoPosition)
    {
        if (Ended) return;
        ImpactSpeed = impactSpeed;
        HazardName = hazardName;
        EgoPosition = egoPosition;
        End(EndReason.Collision, time);
    }
}