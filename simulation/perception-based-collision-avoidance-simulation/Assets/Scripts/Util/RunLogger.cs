using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

[DefaultExecutionOrder(50)]   // after the executor (5) and the ground-truth logger (30)
public class RunLogger : MonoBehaviour
{
    static readonly HashSet<string> SafeUnityArgs = new HashSet<string>
    {
        "-batchmode", "-nographics", "-screen-width", "-screen-height", "-screen-fullscreen", "-logFile", "-projectPath"
    };

    static string SafeCommandLine()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        var kept = new List<string>();
        int omitted = 0;
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (i == 0) { kept.Add(a); continue; }
            bool keep = a.StartsWith("--") || SafeUnityArgs.Contains(a);
            bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("-");
            if (!keep)
            {
                omitted++;
                if (hasValue) i++;          // its value goes with it
                continue;
            }
            kept.Add(a);
            if (hasValue) { kept.Add(args[i + 1]); i++; }
        }
        if (omitted > 0) kept.Add("(+" + omitted + " other Unity arguments omitted)");
        return string.Join(" ", kept);
    }

    public SensorRig rig;
    public CommandExecutor executor;
    public CalibrationProvider calibration;
    public string scenarioName = "dev";

    // true when nothing is left to write: the logs were closed, or never opened (no run folder, or the run ended before READY)
    public bool IsClosed { get { return closed || !opened; } }

    StreamWriter commands, frames;
    readonly List<float> frameMs = new List<float>();
    string startedLocal, startedUtc;
    double firstSim = -1, firstReal, lastSim, lastReal, nextFlush;
    float timeScaleDuringRun = -1f;
    bool opened, calibrationCopied, closed;
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    static string R(double v) { return v.ToString("R", Inv); }

    void Open()
    {
        opened = true;
        startedLocal = System.DateTime.Now.ToString("o");
        startedUtc = System.DateTime.UtcNow.ToString("o");
        File.WriteAllText(Path.Combine(RunFolder.ConfigCopyDir, "sensors_resolved.json"), SensorConfigLoader.ToJson(rig.Config));
        if (ScenarioContext.Loaded && ScenarioContext.ResolvedJson != null)
            File.WriteAllText(Path.Combine(RunFolder.ConfigCopyDir, "scenario_resolved.json"), ScenarioContext.ResolvedJson);
        commands = new StreamWriter(Path.Combine(RunFolder.UnityDir, "commands.csv"));
        commands.WriteLine("t,speed,accel,jerk,driver_accel,applied_accel,rule,data_age_s,object_id,brake,waiting,remaining,odometer");
        frames = new StreamWriter(Path.Combine(RunFolder.UnityDir, "frames.csv"));
        frames.WriteLine("real_s,sim_s,frame_ms");
        WriteRunJson(false, "");
        nextFlush = Time.realtimeSinceStartupAsDouble + 2.0;
        UnityEngine.Debug.Log("RUN LOG " + RunFolder.Dir);
    }

    void Update()
    {
        if (closed) return;
        if (!opened)
        {
            if (RunFolder.Current == RunFolder.State.Unknown) return;           // READY has not arrived yet
            if (RunFolder.Current == RunFolder.State.Disabled) { enabled = false; return; }
            Open();
        }
        if (!calibrationCopied && calibration != null && calibration.Json != null)
        {
            File.WriteAllText(Path.Combine(RunFolder.ConfigCopyDir, "calibration.json"), calibration.Json);
            calibrationCopied = true;
        }
        if (Time.timeScale <= 0f) return;     // frozen: waiting for the processor, or paused

        double real = Time.realtimeSinceStartupAsDouble, sim = SimClock.Now;
        float ms = Time.unscaledDeltaTime * 1000f;
        if (firstSim < 0) { firstSim = sim; firstReal = real; }
        lastSim = sim; lastReal = real;
        frameMs.Add(ms);
        frames.WriteLine(R(real) + "," + R(sim) + "," + ms.ToString("F3", Inv));
    }

    void FixedUpdate()
    {
        if (closed || !opened) return;
        if (timeScaleDuringRun < 0f) timeScaleDuringRun = Time.timeScale;
        if (executor != null)
            commands.WriteLine(R(SimClock.Now) + "," + R(executor.vehicle.Speed) + "," + R(executor.vehicle.Accel) + "," + R(executor.vehicle.Jerk) + "," +
                               R(executor.driver != null ? executor.driver.DesiredAccel : 0f) + "," + R(executor.AppliedAccel) + "," + executor.AppliedRule + "," +
                               R(executor.DataAge) + "," + executor.ObjectId + "," + R(executor.Brake) + "," + (executor.Waiting ? 1 : 0) + "," +
                               R(executor.Remaining) + "," + R(executor.vehicle.Odometer));

        if (RunControl.Ended) { Close(); return; }
        if (Time.realtimeSinceStartupAsDouble >= nextFlush)
        {
            commands.Flush(); frames.Flush();
            nextFlush = Time.realtimeSinceStartupAsDouble + 2.0;
        }
    }

    void OnDestroy() { if (opened && !closed) Close(); }

    void Close()
    {
        closed = true;
        commands.Close(); frames.Close();
        WriteRunJson(true, RunControl.Ended ? RunControl.Reason.ToString() : "Quit");
    }

    void WriteRunJson(bool final, string reason)
    {
        var run = new Dictionary<string, object>
        {
            { "formatVersion", 1 },
            { "runId", RunFolder.RunId },
            { "scenario", ScenarioContext.Loaded ? ScenarioContext.Config.name : scenarioName },
            { "startedLocal", startedLocal },
            { "startedUtc", startedUtc },
            { "sensorConfig", rig.Config.name },
            { "seed", rig.Seed },
            { "seedNote", "every noise generator derives its seed from this one (SeedUtil.Derive with a salt)" },
            { "fixedDeltaTime", Time.fixedDeltaTime },
            { "unityVersion", UnityEngine.Application.unityVersion },
            { "platform", UnityEngine.Application.platform.ToString() },
            { "isEditor", UnityEngine.Application.isEditor },
            { "graphicsDevice", SystemInfo.graphicsDeviceName },
            { "cpu", SystemInfo.processorType },
            { "cpuThreads", SystemInfo.processorCount },
            { "commandLine", SafeCommandLine() },
        };
        if (ScenarioContext.Loaded)
        {
            run["scenarioFile"] = ScenarioContext.SourcePath;
            run["scenarioOverrides"] = ScenarioContext.Overrides;
            run["systemVariant"] = ScenarioContext.Config.systemVariant;
        }
        if (final)
        {
            var end = new Dictionary<string, object>
            {
                { "reason", reason },
                { "simTime", RunControl.Ended ? RunControl.EndTime : SimClock.Now },
                { "timeScaleDuringRun", timeScaleDuringRun },
            };
            if (RunControl.Ended && RunControl.Reason == RunControl.EndReason.Collision)
            {
                end["impactSpeed"] = RunControl.ImpactSpeed;
                end["hazard"] = RunControl.HazardName;
            }
            if (frameMs.Count > 0)
            {
                var sorted = new List<float>(frameMs);
                sorted.Sort();
                double sum = 0;
                foreach (float f in sorted) sum += f;
                end["frames"] = frameMs.Count;
                end["frameMsMean"] = sum / sorted.Count;
                end["frameMsP99"] = sorted[(int)(0.99 * (sorted.Count - 1))];
                end["frameMsMax"] = sorted[sorted.Count - 1];
                if (lastReal > firstReal) end["simToRealRatio"] = (lastSim - firstSim) / (lastReal - firstReal);   // 1 = real time
            }
            if (RunOutcome.Captured) end["outcome"] = RunOutcome.ToDictionary();
            run["end"] = end;
        }
        File.WriteAllText(Path.Combine(RunFolder.Dir, "run.json"), JsonConvert.SerializeObject(run, Formatting.Indented));
    }
}