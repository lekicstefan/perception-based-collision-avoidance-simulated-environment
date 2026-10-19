using System.Collections.Generic;

// How the run ended, for run.json (end.outcome). Filled by the ScenarioManager at the moment the run ends.
public static class RunOutcome
{
    public static bool Captured { get; private set; }
    public static string Reason { get; private set; }
    public static double SimTime { get; private set; }
    public static float FinalSpeedMps { get; private set; }
    public static float RouteProgressM { get; private set; }
    public static float RouteLengthM { get; private set; }

    public static void Reset()
    {
        Captured = false;
        Reason = null;
        SimTime = 0.0;
        FinalSpeedMps = 0f;
        RouteProgressM = 0f;
        RouteLengthM = 0f;
    }

    public static void Capture(string reason, double simTime, float finalSpeedMps, float routeProgressM, float routeLengthM)
    {
        Captured = true;
        Reason = reason;
        SimTime = simTime;
        FinalSpeedMps = finalSpeedMps;
        RouteProgressM = routeProgressM;
        RouteLengthM = routeLengthM;
    }

    public static Dictionary<string, object> ToDictionary()
    {
        var d = new Dictionary<string, object>
        {
            { "reason", Reason },
            { "simTime", SimTime },
            { "finalSpeedMps", FinalSpeedMps },
            { "routeProgressM", RouteProgressM },
            { "routeLengthM", RouteLengthM },
            { "routeFraction", RouteLengthM > 0f ? RouteProgressM / RouteLengthM : 0f },
        };
        if (RunControl.Ended && RunControl.Reason == RunControl.EndReason.Collision)
        {
            d["impactSpeedMps"] = RunControl.ImpactSpeed;
            d["hazard"] = RunControl.HazardName;
        }
        return d;
    }
}