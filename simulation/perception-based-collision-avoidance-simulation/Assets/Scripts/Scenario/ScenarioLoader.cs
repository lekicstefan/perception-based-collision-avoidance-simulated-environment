using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// The scenario of the current run, for the loggers (RunLogger). Set by ScenarioSettings.
public static class ScenarioContext
{
    public static bool Loaded { get; private set; }
    public static ScenarioConfig Config { get; private set; }     // as loaded (after the overrides)
    public static string SourcePath { get; private set; }
    public static string[] Overrides { get; private set; }
    public static string ResolvedJson { get; private set; }       // every value that was actually used; set once the scene has started

    [UnityEngine.RuntimeInitializeOnLoadMethod(UnityEngine.RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetForNewPlaySession() { Loaded = false; Config = null; SourcePath = null; Overrides = null; ResolvedJson = null; }

    public static void SetLoaded(ScenarioConfig config, string sourcePath, string[] overrides)
    {
        Config = config; SourcePath = sourcePath; Overrides = overrides; Loaded = true;
    }

    public static void SetResolved(string json) { ResolvedJson = json; }
}

public static class ScenarioLoader
{
    // A misspelled key is an error. Null values in the file are ignored (the default stays).
    static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
    {
        MissingMemberHandling = MissingMemberHandling.Error,
        NullValueHandling = NullValueHandling.Ignore,
        Formatting = Formatting.Indented
    };

    public static string PathFor(string nameOrPath)
    {
        if (nameOrPath.EndsWith(".json") || Path.IsPathRooted(nameOrPath)) return ConfigPaths.Resolve(nameOrPath);
        return ConfigPaths.Resolve(Path.Combine("scenarios", nameOrPath + ".json"));
    }

    public static List<string> Available()
    {
        var names = new List<string>();
        string dir = ConfigPaths.Resolve("scenarios");
        if (Directory.Exists(dir))
            foreach (string f in Directory.GetFiles(dir, "*.json")) names.Add(Path.GetFileNameWithoutExtension(f));
        names.Sort();
        return names;
    }

    // overrides: "path=value" strings applied to the JSON before it is read, for example driver.cruiseSpeedMps=11.1 or
    // objects.CrossingVehicle.startDelayS=4.2 (a sweep point). The value is read as JSON (number, true, false) or else as text.
    public static ScenarioConfig Load(string nameOrPath, IList<string> overrides, out string sourcePath)
    {
        sourcePath = PathFor(nameOrPath);
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("Scenario file not found: " + sourcePath + ". Available scenarios: " +
                                            (Available().Count > 0 ? string.Join(", ", Available()) : "(none)"));

        JObject root;
        try { root = JObject.Parse(File.ReadAllText(sourcePath)); }
        catch (JsonReaderException e) { throw new ArgumentException("Scenario file " + sourcePath + " is not valid JSON: " + e.Message, e); }

        foreach (string o in overrides) ApplyOverride(root, o);

        ScenarioConfig c;
        try { c = root.ToObject<ScenarioConfig>(JsonSerializer.Create(Settings)); }
        catch (JsonSerializationException e) { throw new ArgumentException("Scenario '" + nameOrPath + "': " + e.Message, e); }
        catch (JsonReaderException e) { throw new ArgumentException("Scenario '" + nameOrPath + "': " + e.Message, e); }

        Validate(c, Path.GetFileNameWithoutExtension(sourcePath));
        return c;
    }

    public static string ToJson(ScenarioConfig c) { return JsonConvert.SerializeObject(c, Settings); }

    static void ApplyOverride(JObject root, string spec)
    {
        int eq = spec.IndexOf('=');
        if (eq <= 0) throw new ArgumentException("--set expects path=value, got '" + spec + "'");
        string path = spec.Substring(0, eq).Trim();
        string raw = spec.Substring(eq + 1).Trim();

        JToken value;
        try { value = JToken.Parse(raw); }
        catch (JsonReaderException) { value = new JValue(raw); }

        string[] parts = path.Split('.');
        JObject cur = root;
        for (int i = 0; i + 1 < parts.Length; i++)
        {
            JToken next = cur[parts[i]];
            if (next == null || next.Type == JTokenType.Null)
            {
                next = new JObject();
                cur[parts[i]] = next;
            }
            else if (!(next is JObject))
                throw new ArgumentException("--set " + spec + ": '" + parts[i] + "' is not a group of settings");
            cur = (JObject)next;
        }
        cur[parts[parts.Length - 1]] = value;
    }

    static void Validate(ScenarioConfig c, string fileStem)
    {
        if (c.schema != 1) throw new ArgumentException("Scenario schema " + c.schema + " is not supported (expected 1)");
        if (string.IsNullOrEmpty(c.name)) throw new ArgumentException("Scenario has no 'name'");
        if (c.name != fileStem) throw new ArgumentException("Scenario name '" + c.name + "' must equal the file name '" + fileStem + "'");
        if (string.IsNullOrEmpty(c.scene)) throw new ArgumentException("Scenario '" + c.name + "' has no 'scene' (the name of its Unity scene)");

        Positive(c, "driver.cruiseSpeedMps", c.driver.cruiseSpeedMps);
        Positive(c, "driver.maxLateralAccelMps2", c.driver.maxLateralAccelMps2);
        Positive(c, "vehicle.wheelbaseM", c.vehicle.wheelbaseM);
        Positive(c, "vehicle.maxAccelMps2", c.vehicle.maxAccelMps2);
        Positive(c, "vehicle.maxBrakeMps2", c.vehicle.maxBrakeMps2);
        NonNegative(c, "vehicle.actuationDelayS", c.vehicle.actuationDelayS);
        NonNegative(c, "executor.capGain", c.executor.capGain);
        NonNegative(c, "environment.fogVisibilityM", c.environment.fogVisibilityM);
        if (c.sensors.seed.HasValue && c.sensors.seed.Value < 0) throw new ArgumentException("Scenario '" + c.name + "': sensors.seed must be 0 or more");
        if (c.objects != null)
            foreach (var kv in c.objects)
            {
                if (kv.Value == null) continue;
                NonNegative(c, "objects." + kv.Key + ".speedMps", kv.Value.speedMps);
                NonNegative(c, "objects." + kv.Key + ".startDelayS", kv.Value.startDelayS);
            }
        if (c.end.timeoutS <= 0f) throw new ArgumentException("Scenario '" + c.name + "': end.timeoutS must be above 0");
        if (c.end.stoppedForS < 0f) throw new ArgumentException("Scenario '" + c.name + "': end.stoppedForS must be 0 or more");
        if (c.end.stoppedSpeedMps < 0f) throw new ArgumentException("Scenario '" + c.name + "': end.stoppedSpeedMps must be 0 or more");
    }

    static void Positive(ScenarioConfig c, string label, float? v)
    {
        if (v.HasValue && !(v.Value > 0f)) throw new ArgumentException("Scenario '" + c.name + "': " + label + " must be above 0 (is " + v.Value + ")");
    }

    static void NonNegative(ScenarioConfig c, string label, float? v)
    {
        if (v.HasValue && !(v.Value >= 0f)) throw new ArgumentException("Scenario '" + c.name + "': " + label + " must be 0 or more (is " + v.Value + ")");
    }
}