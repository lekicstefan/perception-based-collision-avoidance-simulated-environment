using System.Collections.Generic;

// How the run ended, for run.json (end.outcome). Filled by the ScenarioManager at the moment the run ends.
// Also records what the person at the keyboard did to the run: pauses and any change of the time scale.
// A run at a time scale other than 1 is not valid for experiments (both programs run on real wall-clock time, so the processor would be
// too slow or too fast for the simulation), and the outcome says so.
public static class RunOutcome
{
    public static bool Captured { get; private set; }
    public static string Reason { get; private set; }
    public static double SimTime { get; private set; }
    public static float FinalSpeedMps { get; private set; }
    public static float RouteProgressM { get; private set; }
    public static float RouteLengthM { get; private set; }

    public static bool TimeScaleChanged { get; private set; }
    public static float MinTimeScale { get; private set; }
    public static float MaxTimeScale { get; private set; }
    public static int Pauses { get; private set; }
    public static double PausedRealS { get; private set; }

    public static bool ValidForExperiments { get { return !TimeScaleChanged; } }

    // What the moving objects did (start times, speed events, trigger schedule), from WaypointMover.CollectReports().
    public static object Objects { get; private set; }

    public static void Reset()
    {
        Captured = false;
        Reason = null;
        SimTime = 0.0;
        FinalSpeedMps = 0f;
        RouteProgressM = 0f;
        RouteLengthM = 0f;
        TimeScaleChanged = false;
        MinTimeScale = 1f;
        MaxTimeScale = 1f;
        Pauses = 0;
        PausedRealS = 0.0;
        Objects = null;
    }

    public static void SetObjects(object reports) { Objects = reports; }

    public static void Capture(string reason, double simTime, float finalSpeedMps, float routeProgressM, float routeLengthM)
    {
        Captured = true;
        Reason = reason;
        SimTime = simTime;
        FinalSpeedMps = finalSpeedMps;
        RouteProgressM = routeProgressM;
        RouteLengthM = routeLengthM;
    }

    // The simulation ran (not paused) at this time scale. Returns true the first time a scale other than 1 is seen.
    public static bool NoteTimeScale(float scale)
    {
        if (scale < MinTimeScale) MinTimeScale = scale;
        if (scale > MaxTimeScale) MaxTimeScale = scale;
        if (System.Math.Abs(scale - 1f) <= 0.001f) return false;
        bool first = !TimeScaleChanged;
        TimeScaleChanged = true;
        return first;
    }

    public static void NotePause(double realSeconds)
    {
        Pauses++;
        PausedRealS += realSeconds;
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
            { "validForExperiments", ValidForExperiments },
            { "timeScaleChanged", TimeScaleChanged },
            { "minTimeScale", MinTimeScale },
            { "maxTimeScale", MaxTimeScale },
            { "pauses", Pauses },
            { "pausedRealS", PausedRealS },
        };
        if (Objects != null) d["objects"] = Objects;
        if (RunControl.Ended && RunControl.Reason == RunControl.EndReason.Collision)
        {
            d["impactSpeedMps"] = RunControl.ImpactSpeed;
            d["hazard"] = RunControl.HazardName;
        }
        return d;
    }
}