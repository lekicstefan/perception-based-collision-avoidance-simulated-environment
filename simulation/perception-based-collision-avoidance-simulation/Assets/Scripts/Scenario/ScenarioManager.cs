using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;
using static System.Net.Mime.MediaTypeNames;

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
//   Esc back to the menu   Space pause / resume   E end the run   R restart   Up / Right faster   Down / Left slower   0 back to 1x
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
    string pendingScene;                  // a scene to load when the run has ended (the menu)
    float stoppedFor;
    string endText = "";
    string endDetail = "";                // the same for the HUD: time, impact speed
    int scaleIndex = 2;
    double pauseStartReal;

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
        endDetail = "t " + RunControl.EndTime.ToString("F2") + " s";
        if (RunControl.Reason == RunControl.EndReason.Collision)
            endDetail += "    impact " + (RunControl.ImpactSpeed * 3.6f).ToString("F1") + " km/h with " + RunControl.HazardName;
        else endDetail += "    " + (vehicle.Speed * 3.6f).ToString("F0") + " km/h at the end";
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

        if (pendingScene != null)
        {
            LoadPending();
            yield break;
        }
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

    // Ends the run if one is going and goes to the scenario menu scene (ScenarioContext.MenuScene, in the Build Settings).
    public void ToMenu()
    {
        if (pendingScene != null || restartRequested) return;
        if (!UnityEngine.Application.CanStreamedLevelBeLoaded(ScenarioContext.MenuScene))
        {
            UnityEngine.Debug.LogWarning("SCENARIO MANAGER: there is no scene '" + ScenarioContext.MenuScene + "' in the Build Settings, so there is no menu to go to");
            return;
        }
        pendingScene = ScenarioContext.MenuScene;
        if (State == RunState.Ended) LoadPending();
        else if (State != RunState.Ending) Finish(RunControl.EndReason.Manual);   // Ending: the finish routine loads it when it is done
    }

    void LoadPending()
    {
        string scene = pendingScene;
        pendingScene = null;
        UnityEngine.Debug.Log("MENU | loading scene '" + scene + "'");
        SceneManager.LoadScene(scene);
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
        switch (e.keyCode)
        {
            case KeyCode.Escape: ToMenu(); break;
            case KeyCode.Space: TogglePause(); break;
            case KeyCode.E: EndRun(); break;
            case KeyCode.R: Restart(); break;
            case KeyCode.UpArrow:
            case KeyCode.RightArrow: ChangeSpeed(1); break;
            case KeyCode.DownArrow:
            case KeyCode.LeftArrow: ChangeSpeed(-1); break;
            case KeyCode.Alpha0:
            case KeyCode.Keypad0: ResetSpeed(); break;
            default: return;
        }
        e.Use();
    }

    void OnGUI()
    {
        UiTheme.Begin();
        DrawUi();
        UiTheme.End();
    }

    void DrawUi()
    {
        Event e = Event.current;
        if (hotkeys && e.type == EventType.KeyDown && GUIUtility.keyboardControl == 0 && !e.control && !e.alt) HandleKey(e);

        if (!showControls || !ViewControls.HudVisible) return;
        UiTheme.Ensure();

        float h = UiTheme.PanelHeight(3);
        Rect inner = UiTheme.Panel(new Rect(12f, UiTheme.H - h - 12f, 580f, h));

        // scenario name and state
        Rect[] c = UiTheme.Cols(UiTheme.NextRow(ref inner), 0f, 190f);
        GUI.Label(c[0], ScenarioContext.Loaded ? ScenarioContext.Config.name : "(no scenario)", UiTheme.Name);
        switch (State)
        {
            case RunState.WaitingForProcessor: UiTheme.Status(c[1], "Waiting for processor", UiTheme.Tone.Warn); break;
            case RunState.Running:
                if (Paused) UiTheme.Status(c[1], "Paused", UiTheme.Tone.Warn); else UiTheme.Status(c[1], "Running", UiTheme.Tone.Ok);
                break;
            default: UiTheme.Status(c[1], RunControl.Reason.ToString(), ReasonTone(RunControl.Reason)); break;
        }

        // numbers
        bool invalid = RunOutcome.TimeScaleChanged;
        c = invalid ? UiTheme.Cols(UiTheme.NextRow(ref inner), 0f, 190f) : UiTheme.Cols(UiTheme.NextRow(ref inner), 0f);
        if (State == RunState.Running)
            GUI.Label(c[0], "t " + SimClock.Now.ToString("F1") + " s   " + (vehicle != null ? (vehicle.Speed * 3.6f).ToString("F0") : "0") + " km/h   x" + TimeScale.ToString("0.##"), UiTheme.Value);
        else if (State == RunState.Ended)
            GUI.Label(c[0], endDetail, UiTheme.Value);
        if (invalid) UiTheme.Status(c[1], "Not valid for experiments", UiTheme.Tone.Bad);

        // buttons
        // the first button is as wide for "Resume" as for "Pause"
        c = UiTheme.Cols(UiTheme.NextRow(ref inner), 118f, 70f, 92f, 90f, 0f, 30f, 60f, 30f);
        GUI.enabled = State == RunState.Running;
        if (UiTheme.Press(c[0], Paused ? "Resume  SPACE" : "Pause  SPACE")) TogglePause();
        if (UiTheme.Press(c[1], "End  E")) EndRun();
        GUI.enabled = true;
        if (UiTheme.Press(c[2], "Restart  R")) Restart();
        if (UiTheme.Press(c[3], "Menu  ESC")) ToMenu();
        GUI.enabled = State == RunState.Running;
        if (UiTheme.Press(c[5], "<")) ChangeSpeed(-1);
        if (UiTheme.Press(c[6], "x" + Scales[scaleIndex] + "  0")) ResetSpeed();
        if (UiTheme.Press(c[7], ">")) ChangeSpeed(1);
        GUI.enabled = true;
    }

    static UiTheme.Tone ReasonTone(RunControl.EndReason r)
    {
        switch (r)
        {
            case RunControl.EndReason.Collision: return UiTheme.Tone.Bad;
            case RunControl.EndReason.Timeout: return UiTheme.Tone.Warn;
            case RunControl.EndReason.RouteFinished:
            case RunControl.EndReason.Stopped: return UiTheme.Tone.Ok;
            default: return UiTheme.Tone.Neutral;
        }
    }
}