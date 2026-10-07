using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

// Run metadata and the Unity-side logs of a run (step 4.7): run.json, copies of the configuration used, the commands
// as applied by the executor, and frame times. Ground truth is written into the same folder by GroundTruthLogger.
[DefaultExecutionOrder(50)]   // after the executor (5) and the ground-truth logger (30)
public class RunLogger : MonoBehaviour
{
    public SensorRig rig;
    public CommandExecutor executor;
    public CalibrationProvider calibration;
    public string scenarioName = "dev";

    StreamWriter commands, frames;
    readonly List<float> frameMs = new List<float>();
    string startedLocal, startedUtc;
    double firstSim = -1, firstReal, lastSim, lastReal, nextFlush;
    float timeScaleDuringRun = -1f;
    bool calibrationCopied, closed;
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    static string R(double v) { return v.ToString("R", Inv); }

    void Start()
    {
        startedLocal = System.DateTime.Now.ToString("o");
        startedUtc = System.DateTime.UtcNow.ToString("o");
        File.WriteAllText(Path.Combine(RunFolder.ConfigCopyDir, "sensors_resolved.json"), SensorConfigLoader.ToJson(rig.Config));
        commands = new StreamWriter(Path.Combine(RunFolder.UnityDir, "commands.csv"));
        commands.WriteLine("t,status,rule,last_id,applied_accel,remaining");
        frames = new StreamWriter(Path.Combine(RunFolder.UnityDir, "frames.csv"));
        frames.WriteLine("real_s,sim_s,frame_ms");
        WriteRunJson(false, "");
        nextFlush = Time.realtimeSinceStartupAsDouble + 2.0;
        UnityEngine.Debug.Log("RUN LOG " + RunFolder.Dir);
    }

    void Update()
    {
        if (closed) return;
        if (!calibrationCopied && calibration != null && calibration.Json != null)
        {
            File.WriteAllText(Path.Combine(RunFolder.ConfigCopyDir, "calibration.json"), calibration.Json);
            calibrationCopied = true;
        }
        if (Time.timeScale <= 0f) return;     // frozen: waiting for the processor, or paused

        double real = Time.realtimeSinceStartupAsDouble, sim = Time.fixedTimeAsDouble;
        float ms = Time.unscaledDeltaTime * 1000f;
        if (firstSim < 0) { firstSim = sim; firstReal = real; }
        lastSim = sim; lastReal = real;
        frameMs.Add(ms);
        frames.WriteLine(R(real) + "," + R(sim) + "," + ms.ToString("F3", Inv));
    }

    void FixedUpdate()
    {
        if (closed) return;
        if (timeScaleDuringRun < 0f) timeScaleDuringRun = Time.timeScale;
        if (executor != null)
            commands.WriteLine(R(Time.fixedTimeAsDouble) + "," + executor.Status + "," + executor.AppliedRule + "," +
                               executor.LastId + "," + R(executor.AppliedAccel) + "," + R(executor.Remaining));

        if (RunControl.Ended) { Close(); return; }
        if (Time.realtimeSinceStartupAsDouble >= nextFlush)
        {
            commands.Flush(); frames.Flush();
            nextFlush = Time.realtimeSinceStartupAsDouble + 2.0;
        }
    }

    void OnDestroy() { if (!closed) Close(); }

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
            { "scenario", scenarioName },
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
            { "commandLine", string.Join(" ", System.Environment.GetCommandLineArgs()) },
        };
        if (final)
        {
            var end = new Dictionary<string, object>
            {
                { "reason", reason },
                { "simTime", RunControl.Ended ? RunControl.EndTime : Time.fixedTimeAsDouble },
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
            run["end"] = end;
        }
        File.WriteAllText(Path.Combine(RunFolder.Dir, "run.json"), JsonConvert.SerializeObject(run, Formatting.Indented));
    }
}