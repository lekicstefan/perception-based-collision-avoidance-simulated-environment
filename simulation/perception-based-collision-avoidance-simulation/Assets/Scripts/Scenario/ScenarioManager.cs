using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

// Owns the life of one run in a scenario scene: start, the end conditions, the outcome, restart, pause and the time scale. One per scene.
//
// A run starts when the processor answers READY (NetworkHost.Started). It ends at the first of:
//   collision        (CollisionDetector ends the run and freezes the car)
//   route finished   (the driver reached the end of its route and the car stands still)
//   stopped          (the car has driven and then stood still for end.stoppedForS)
//   timeout          (end.timeoutS of simulation time)
//   manual           (restart, or "End run")
// The conditions come from the scenario file (end group, ScenarioContext). At the end the manager tells the processor (END_OF_RUN), records the
// outcome in run.json, waits until the logs are closed and then freezes the simulation. Restart reloads the scene, which starts a new run with
// a new handshake and a new run folder (the processor must run with --loop for that).
//
// Pause freezes the simulation (Time.timeScale 0). The network thread keeps sending the calibration message every second, which the processor
// takes as a heartbeat, so a pause of any length does not break the link. Commands expire in simulation time, so they do not expire in a pause.
// The speed keys change Time.timeScale (0.25x ... 4x). Both programs run on real wall-clock time, so a run at any other speed than 1x is
// marked as not valid for experiments in run.json (end.outcome.validForExperiments). Every new run starts at 1x.
//
// Keys (click the Game view first; ignored while a text field has the keyboard):
//   R restart   P pause / resume   X end the run   + faster   - slower   0 back to 1x
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

    static readonly float[] Scales = { 0.25f, 0.5f, 1f, 2f, 4f };

    public RunState State { get; private set; }
    public bool Paused { get; private set; }
    public float TimeScale { get { return Scales[scaleIndex]; } }

    EndParams end = new EndParams();
    bool hasMoved, restartRequested;
    float stoppedFor;
    string endText = "";
    int scaleIndex = 2;
    double pauseStartReal;
    GUIStyle warnStyle;

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

        if (State == RunState.Running && !Paused)
        {
            if (RunOutcome.NoteTimeScale(Time.timeScale))
                UnityEngine.Debug.LogWarning("TIME SCALE " + Time.timeScale.ToString("0.##") + "x | this run is NOT valid for experiments (both programs run on real wall-clock time)");
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

        // The world has to run for a step or two so that the run logger can close its files: back to normal speed, and out of a pause.
        EndPause();
        Time.timeScale = 1f;

        RunControl.End(reason, SimClock.Now);    // no effect when the collision detector already set it
        float progress = driver != null ? driver.S : 0f;
        float length = driver != null && driver.Track != null ? driver.Track.Length : 0f;
        RunOutcome.Capture(RunControl.Reason.ToString(), RunControl.EndTime, vehicle.Speed, progress, length);
        RunOutcome.SetObjects(WaypointMover.CollectReports());

        endText = RunControl.Reason + " at " + RunControl.EndTime.ToString("F2") + " s";
        if (RunControl.Reason == RunControl.EndReason.Collision)
            endText += " | impact " + RunControl.ImpactSpeed.ToString("F2") + " m/s (" + (RunControl.ImpactSpeed * 3.6f).ToString("F1") + " km/h) with " + RunControl.HazardName;
        if (!RunOutcome.ValidForExperiments) endText += " | NOT VALID FOR EXPERIMENTS (time scale changed)";
        UnityEngine.Debug.Log("RUN END | " + endText + " | speed " + vehicle.Speed.ToString("F2") + " m/s | route " + progress.ToString("F1") + " of " + length.ToString("F1") + " m" +
                              (RunOutcome.Pauses > 0 ? " | paused " + RunOutcome.Pauses + " time(s), " + RunOutcome.PausedRealS.ToString("F1") + " s" : ""));

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

    // ---------------------------------------------------------------- controls

    // Ends the run if one is going and starts a new one in a fresh scene.
    public void Restart()
    {
        if (restartRequested) return;
        restartRequested = true;
        if (State == RunState.Ended) Reload();
        else if (State != RunState.Ending) Finish(RunControl.EndReason.Manual);   // Ending: the finish routine reloads when it is done
    }

    // Ends the run now (reason Manual) and leaves the scene frozen. Restart starts the next one.
    public void EndRun()
    {
        if (State == RunState.Running) Finish(RunControl.EndReason.Manual);
    }

    public void TogglePause()
    {
        if (State != RunState.Running) return;
        if (Paused)
        {
            EndPause();
            Time.timeScale = Scales[scaleIndex];
            UnityEngine.Debug.Log("RESUME | t = " + SimClock.Now.ToString("F2") + " s");
        }
        else
        {
            Paused = true;
            pauseStartReal = Time.realtimeSinceStartupAsDouble;
            Time.timeScale = 0f;
            UnityEngine.Debug.Log("PAUSE | t = " + SimClock.Now.ToString("F2") + " s");
        }
    }

    void EndPause()
    {
        if (!Paused) return;
        Paused = false;
        RunOutcome.NotePause(Time.realtimeSinceStartupAsDouble - pauseStartReal);
    }

    public void ChangeSpeed(int steps)
    {
        if (State != RunState.Running) return;
        scaleIndex = Mathf.Clamp(scaleIndex + steps, 0, Scales.Length - 1);
        if (!Paused) Time.timeScale = Scales[scaleIndex];
    }

    public void ResetSpeed()
    {
        if (State != RunState.Running) return;
        scaleIndex = 2;
        if (!Paused) Time.timeScale = Scales[scaleIndex];
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

    // ---------------------------------------------------------------- keys and HUD

    void HandleKey(Event e)
    {
        if (e.keyCode == KeyCode.R) { Restart(); e.Use(); }
        else if (e.keyCode == KeyCode.P) { TogglePause(); e.Use(); }
        else if (e.keyCode == KeyCode.X) { EndRun(); e.Use(); }
        else if (e.keyCode == KeyCode.Alpha0 || e.keyCode == KeyCode.Keypad0) { ResetSpeed(); e.Use(); }
        else if (e.keyCode == KeyCode.None)
        {
            // Symbols are read from the typed character, so they work on every keyboard layout (a plus or a minus is not at the same place everywhere).
            char c = e.character;
            if (c == '+' || c == '=') { ChangeSpeed(1); e.Use(); }
            else if (c == '-' || c == '_') { ChangeSpeed(-1); e.Use(); }
        }
    }

    void OnGUI()
    {
        Event e = Event.current;
        if (hotkeys && e.type == EventType.KeyDown && GUIUtility.keyboardControl == 0 && !e.control && !e.alt) HandleKey(e);

        if (!showControls || !ViewControls.HudVisible) return;

        if (warnStyle == null)
        {
            warnStyle = new GUIStyle(GUI.skin.label);
            warnStyle.normal.textColor = new Color(1f, 0.35f, 0.3f);
            warnStyle.fontStyle = FontStyle.Bold;
        }

        GUILayout.BeginArea(new Rect(10f, Screen.height - 132f, 640f, 96f), GUI.skin.box);
        string scenario = ScenarioContext.Loaded ? ScenarioContext.Config.name : "(no scenario)";
        string status;
        switch (State)
        {
            case RunState.WaitingForProcessor: status = "waiting for the processor"; break;
            case RunState.Running:
                status = (Paused ? "PAUSED" : "running") + ", t = " + SimClock.Now.ToString("F1") + " s, " +
                         (vehicle != null ? (vehicle.Speed * 3.6f).ToString("F0") : "0") + " km/h, speed " + TimeScale.ToString("0.##") + "x";
                break;
            default: status = "ENDED: " + endText; break;
        }
        GUILayout.Label("Scenario " + scenario + " | " + status);
        GUILayout.Label(RunOutcome.TimeScaleChanged ? "Time scale was changed: this run is NOT valid for experiments" : " ", warnStyle);

        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Restart (R)", GUILayout.Width(100f))) Restart();
        GUI.enabled = State == RunState.Running;
        if (GUILayout.Button(Paused ? "Resume (P)" : "Pause (P)", GUILayout.Width(100f))) TogglePause();
        if (GUILayout.Button("End run (X)", GUILayout.Width(100f))) EndRun();
        GUILayout.Space(12f);
        if (GUILayout.Button("-", GUILayout.Width(30f))) ChangeSpeed(-1);
        if (GUILayout.Button("1x (0)", GUILayout.Width(60f))) ResetSpeed();
        if (GUILayout.Button("+", GUILayout.Width(30f))) ChangeSpeed(1);
        GUI.enabled = true;
        GUILayout.EndHorizontal();
        GUILayout.EndArea();
    }
}