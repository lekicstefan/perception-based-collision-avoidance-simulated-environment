using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public static class ScenarioContext
{
    public static bool Loaded { get; private set; }
    public static ScenarioConfig Config { get; private set; }     // as loaded (after the overrides)
    public static string SourcePath { get; private set; }
    public static string[] Overrides { get; private set; }
    public static string ResolvedJson { get; private set; }       // every value that was actually used; set once the scene has started

    public const string MenuScene = "Menu";                        // the scene with the scenario menu (ScenarioMenu)
    public static string Pick;                                     // the scenario chosen in the menu: wins over --scenario and the Scenario field, until the play session ends

    [UnityEngine.RuntimeInitializeOnLoadMethod(UnityEngine.RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetForNewPlaySession() { Loaded = false; Config = null; SourcePath = null; Overrides = null; ResolvedJson = null; Pick = null; }

    public static void SetLoaded(ScenarioConfig config, string sourcePath, string[] overrides)
    {
        Config = config; SourcePath = sourcePath; Overrides = overrides; Loaded = true;
    }

    public static void SetResolved(string json) { ResolvedJson = json; }
}

public static class ScenarioLoader
{
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

        // A part that is a number steps into an entry of an existing list (objects.Ped.events.0.speedMps=3). Missing groups of settings are created.
        string[] parts = path.Split('.');
        JToken cur = root;
        for (int i = 0; i + 1 < parts.Length; i++)
        {
            string part = parts[i];
            JToken next;
            if (cur is JArray)
            {
                var arr = (JArray)cur;
                int idx;
                if (!int.TryParse(part, out idx) || idx < 0 || idx >= arr.Count)
                    throw new ArgumentException("--set " + spec + ": '" + part + "' is not the number of an existing entry (the list has " + arr.Count + ", counted from 0)");
                next = arr[idx];
            }
            else
            {
                var obj = (JObject)cur;
                next = obj[part];
                if (next == null || next.Type == JTokenType.Null)
                {
                    next = new JObject();
                    obj[part] = next;
                }
            }
            if (!(next is JObject) && !(next is JArray))
                throw new ArgumentException("--set " + spec + ": '" + part + "' is not a group of settings");
            cur = next;
        }

        string last = parts[parts.Length - 1];
        if (cur is JArray)
        {
            var arr = (JArray)cur;
            int idx;
            if (!int.TryParse(last, out idx) || idx < 0 || idx >= arr.Count)
                throw new ArgumentException("--set " + spec + ": '" + last + "' is not the number of an existing entry (the list has " + arr.Count + ", counted from 0)");
            arr[idx] = value;
        }
        else ((JObject)cur)[last] = value;
    }

    static void Validate(ScenarioConfig c, string fileStem)
    {
        if (c.schema != 1) throw new ArgumentException("Scenario schema " + c.schema + " is not supported (expected 1)");
        if (string.IsNullOrEmpty(c.name)) throw new ArgumentException("Scenario has no 'name'");
        if (c.name != fileStem) throw new ArgumentException("Scenario name '" + c.name + "' must equal the file name '" + fileStem + "'");
        if (string.IsNullOrEmpty(c.scene)) throw new ArgumentException("Scenario '" + c.name + "' has no 'scene' (the name of its Unity scene)");

        Positive(c, "driver.cruiseSpeedMps", c.driver.cruiseSpeedMps);
        NonNegative(c, "driver.initialSpeedMps", c.driver.initialSpeedMps);
        Positive(c, "driver.maxLateralAccelMps2", c.driver.maxLateralAccelMps2);
        Positive(c, "vehicle.wheelbaseM", c.vehicle.wheelbaseM);
        Positive(c, "vehicle.maxAccelMps2", c.vehicle.maxAccelMps2);
        Positive(c, "vehicle.maxBrakeMps2", c.vehicle.maxBrakeMps2);
        NonNegative(c, "vehicle.actuationDelayS", c.vehicle.actuationDelayS);
        NonNegative(c, "environment.fogVisibilityM", c.environment.fogVisibilityM);
        if (c.sensors.seed.HasValue && c.sensors.seed.Value < 0) throw new ArgumentException("Scenario '" + c.name + "': sensors.seed must be 0 or more");
        if (c.objects != null)
            foreach (var kv in c.objects)
                if (kv.Value != null) ValidateObject(c, kv.Key, kv.Value);
        if (c.end.timeoutS <= 0f) throw new ArgumentException("Scenario '" + c.name + "': end.timeoutS must be above 0");
        if (c.end.stoppedForS < 0f) throw new ArgumentException("Scenario '" + c.name + "': end.stoppedForS must be 0 or more");
        if (c.end.stoppedSpeedMps < 0f) throw new ArgumentException("Scenario '" + c.name + "': end.stoppedSpeedMps must be 0 or more");
    }

    static void ValidateObject(ScenarioConfig c, string name, ObjectParams o)
    {
        string p = "objects." + name;
        NonNegative(c, p + ".speedMps", o.speedMps);
        NonNegative(c, p + ".startDelayS", o.startDelayS);

        if (o.trigger != null)
        {
            if (!o.trigger.arrivalOffsetS.HasValue)
                throw new ArgumentException("Scenario '" + c.name + "': " + p + ".trigger needs arrivalOffsetS");
            if (o.startDelayS.HasValue)
                throw new ArgumentException("Scenario '" + c.name + "': " + p + " has both startDelayS and a trigger. With a trigger the start time is computed, remove startDelayS");
            NonNegative(c, p + ".trigger.egoRouteM", o.trigger.egoRouteM);
            NonNegative(c, p + ".trigger.objectPathM", o.trigger.objectPathM);
        }

        if (o.events != null)
            for (int i = 0; i < o.events.Count; i++)
            {
                MotionEventParams e = o.events[i];
                string ep = p + ".events[" + i + "]";
                if (e == null) throw new ArgumentException("Scenario '" + c.name + "': " + ep + " is empty");
                int when = (e.atTimeS.HasValue ? 1 : 0) + (e.atPathM.HasValue ? 1 : 0) + (e.afterPreviousS.HasValue ? 1 : 0);
                if (when != 1)
                    throw new ArgumentException("Scenario '" + c.name + "': " + ep + " needs exactly one of atTimeS, atPathM, afterPreviousS (has " + when + ")");
                if (!e.speedMps.HasValue)
                    throw new ArgumentException("Scenario '" + c.name + "': " + ep + " needs speedMps");
                NonNegative(c, ep + ".atTimeS", e.atTimeS);
                NonNegative(c, ep + ".atPathM", e.atPathM);
                NonNegative(c, ep + ".afterPreviousS", e.afterPreviousS);
                NonNegative(c, ep + ".speedMps", e.speedMps);
                NonNegative(c, ep + ".accelMps2", e.accelMps2);
            }
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