using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

// Owns the life of one run in a scenario scene: start, the end conditions, the outcome, and restart. One per scene.
//
// A run starts when the processor answers READY (NetworkHost.Started). It ends at the first of:
//   collision        (CollisionDetector ends the run and freezes the car)
//   route finished   (the driver reached the end of its route and the car stands still)
//   stopped          (the car has driven and then stood still for end.stoppedForS)
//   timeout          (end.timeoutS of simulation time)
//   manual           (restart)
// The conditions come from the scenario file (end group, ScenarioContext). At the end the manager tells the processor (END_OF_RUN), records the
// outcome in run.json, waits until the logs are closed and then freezes the simulation. Restart reloads the scene, which starts a new run with
// a new handshake and a new run folder (the processor must run with --loop for that).
//
// The scene must be listed in the Build Settings for a restart to work (File > Build Profiles > Scene List).
[DefaultExecutionOrder(-400)]   // before ScenarioSettings (-300) and everything else: resets the per-run statics first
public class ScenarioManager : MonoBehaviour
{
    public enum RunState { WaitingForProcessor, Running, Ending, Ended }

    [Header("Wiring (found in the scene when left empty)")]
    public Vehicle vehicle;
    public RouteDriver driver;
    public NetworkHost network;
    public RunLogger runLogger;
    public GroundTruthLogger groundTruth;

    [Header("Controls")]
    public bool showControls = true;
    public bool hotkeys = true;

    public RunState State { get; private set; }

    EndParams end = new EndParams();
    bool hasMoved, restartRequested;
    float stoppedFor;
    string endText = "";

    void Awake()
    {
        // The statics of the previous run (this scene may be a reload) and Unity's global clock, which never goes back to 0.
        Time.timeScale = 1f;
        SimClock.StartNow();
        RunFolder.Reset();
        RunControl.Reset();
        RunOutcome.Reset();
        Hazard.ResetIds();

        if (vehicle == null) vehicle = FindFirstObjectByType<Vehicle>();
        if (driver == null) driver = FindFirstObjectByType<RouteDriver>();
        if (network == null) network = FindFirstObjectByType<NetworkHost>();
        if (runLogger == null) runLogger = FindFirstObjectByType<RunLogger>();
        if (groundTruth == null) groundTruth = FindFirstObjectByType<GroundTruthLogger>();
        State = RunState.WaitingForProcessor;
    }

    void Start()
    {
        if (ScenarioContext.Loaded) end = ScenarioContext.Config.end;
        else UnityEngine.Debug.LogWarning("SCENARIO MANAGER: no scenario loaded (add a ScenarioSettings component), using the default end conditions");
        if (!end.onCollision)
            UnityEngine.Debug.LogWarning("SCENARIO MANAGER: end.onCollision=false is not supported, a collision always ends the run (the car is frozen at the impact)");
    }

    void Update()
    {
        if (State == RunState.WaitingForProcessor && (network == null || !network.waitForProcessor || network.Started))
        {
            State = RunState.Running;
            UnityEngine.Debug.Log("RUN START | scenario " + (ScenarioContext.Loaded ? ScenarioContext.Config.name : "(none)") + " | run folder " +
                                  (RunFolder.Current == RunFolder.State.Enabled ? RunFolder.Dir : "(none)"));
        }
    }

    void FixedUpdate()
    {
        if (State != RunState.Running) return;

        if (RunControl.Ended)                    // the collision detector ended it
        {
            Finish(RunControl.Reason);
            return;
        }

        double t = SimClock.Now;
        float speed = vehicle.Speed;
        if (speed > 0.5f) hasMoved = true;

        if (end.onRouteFinished && driver != null && driver.Finished && speed <= end.stoppedSpeedMps)
        {
            Finish(RunControl.EndReason.RouteFinished);
            return;
        }

        if (end.stoppedForS > 0f && hasMoved)
        {
            stoppedFor = speed <= end.stoppedSpeedMps ? stoppedFor + Time.fixedDeltaTime : 0f;
            if (stoppedFor >= end.stoppedForS)
            {
                Finish(RunControl.EndReason.Stopped);
                return;
            }
        }

        if (t >= end.timeoutS) Finish(RunControl.EndReason.Timeout);
    }

    void Finish(RunControl.EndReason reason)
    {
        if (State == RunState.Ending || State == RunState.Ended) return;
        State = RunState.Ending;

        RunControl.End(reason, SimClock.Now);    // no effect when the collision detector already set it
        float progress = driver != null ? driver.S : 0f;
        float length = driver != null && driver.Track != null ? driver.Track.Length : 0f;
        RunOutcome.Capture(RunControl.Reason.ToString(), RunControl.EndTime, vehicle.Speed, progress, length);

        endText = RunControl.Reason + " at " + RunControl.EndTime.ToString("F2") + " s";
        if (RunControl.Reason == RunControl.EndReason.Collision)
            endText += " | impact " + RunControl.ImpactSpeed.ToString("F2") + " m/s (" + (RunControl.ImpactSpeed * 3.6f).ToString("F1") + " km/h) with " + RunControl.HazardName;
        UnityEngine.Debug.Log("RUN END | " + endText + " | speed " + vehicle.Speed.ToString("F2") + " m/s | route " + progress.ToString("F1") + " of " + length.ToString("F1") + " m");

        if (network != null) network.RequestEndOfRun(RunControl.EndTime);
        StartCoroutine(FinishRoutine());
    }

    // The run logger closes its files one physics step after the end, so the world keeps running until it has.
    IEnumerator FinishRoutine()
    {
        float waited = 0f;
        while (runLogger != null && !runLogger.IsClosed && waited < 2f)
        {
            waited += Time.unscaledDeltaTime;
            yield return null;
        }
        if (groundTruth != null) groundTruth.Flush();
        Time.timeScale = 0f;
        State = RunState.Ended;

        if (restartRequested)
        {
            Reload();
            yield break;
        }
        if (Cli.Has("--quit-on-end")) Quit();
    }

    // Ends the run if one is going and starts a new one in a fresh scene.
    public void Restart()
    {
        if (restartRequested) return;
        restartRequested = true;
        if (State == RunState.Ended) Reload();
        else if (State != RunState.Ending) Finish(RunControl.EndReason.Manual);   // Ending: the finish routine reloads when it is done
    }

    void Reload()
    {
        Scene s = SceneManager.GetActiveScene();
        if (s.buildIndex < 0)
        {
            restartRequested = false;
            UnityEngine.Debug.LogError("SCENARIO MANAGER: the scene '" + s.name + "' is not in the Build Settings, so it cannot be restarted. Add it under File > Build Profiles > Scene List.");
            return;
        }
        UnityEngine.Debug.Log("RESTART | reloading scene '" + s.name + "'");
        SceneManager.LoadScene(s.buildIndex);
    }

    static void Quit()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        UnityEngine.Application.Quit();
#endif
    }

    void OnGUI()
    {
        if (!showControls) return;

        Event e = Event.current;
        if (hotkeys && e.type == EventType.KeyDown && e.keyCode == KeyCode.R && GUIUtility.keyboardControl == 0)
        {
            Restart();
            e.Use();
        }

        GUILayout.BeginArea(new Rect(10f, Screen.height - 74f, 560f, 64f), GUI.skin.box);
        string scenario = ScenarioContext.Loaded ? ScenarioContext.Config.name : "(no scenario)";
        string status;
        switch (State)
        {
            case RunState.WaitingForProcessor: status = "waiting for the processor"; break;
            case RunState.Running: status = "running, t = " + SimClock.Now.ToString("F1") + " s, " + (vehicle != null ? (vehicle.Speed * 3.6f).ToString("F0") : "0") + " km/h"; break;
            default: status = "ENDED: " + endText; break;
        }
        GUILayout.Label("Scenario " + scenario + " | " + status);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Restart (R)", GUILayout.Width(110f))) Restart();
        GUILayout.EndHorizontal();
        GUILayout.EndArea();
    }
}