using UnityEngine;

// The simulation clock of a run: simulation seconds since the run's scene was loaded. Every timestamp (sensors, pose, commands, logs,
// END_OF_RUN) uses this, so a run that follows a restart looks exactly like one in a fresh process, starting at 0.
// Unity's own Time.fixedTimeAsDouble never goes back to 0 while the application runs (it keeps counting across scene loads),
// so the ScenarioManager records its value when a scene loads and this class subtracts it.
//
// Exception: when Play mode is stopped in the editor, Unity sets its clock back to 0 before the OnDestroy methods run. Subtracting the
// origin would then give a negative time (for example -34 s in END_OF_RUN), so a negative result returns the last valid value instead.
public static class SimClock
{
    static double origin;
    static double last;

    public static double Origin { get { return origin; } }

    public static double Now
    {
        get
        {
            double t = Time.fixedTimeAsDouble - origin;
            if (t < 0.0) return last;       // teardown: Unity has already reset its clock
            last = t;
            return t;
        }
    }

    // Called when a scene loads. While Time.timeScale is 0 (waiting for the processor) fixed time does not advance, so this is the run's time 0.
    public static void StartNow() { origin = Time.fixedTimeAsDouble; last = 0.0; }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetForNewPlaySession() { origin = 0.0; last = 0.0; }
}