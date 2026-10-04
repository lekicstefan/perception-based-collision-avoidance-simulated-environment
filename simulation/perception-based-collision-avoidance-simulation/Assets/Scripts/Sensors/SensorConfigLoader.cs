using System.IO;
using Newtonsoft.Json;

public static class SensorConfigLoader
{
    // Default object handling reuses the objects created by the field initializers, so nested partial
    // files keep the other defaults. Arrays are replaced (attributes on the array fields).
    static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
    {
        MissingMemberHandling = MissingMemberHandling.Error,   // a misspelled key is an error
        Formatting = Formatting.Indented
    };

    public static string PathFor(string nameOrPath)
    {
        if (nameOrPath.EndsWith(".json") || Path.IsPathRooted(nameOrPath)) return ConfigPaths.Resolve(nameOrPath);
        return ConfigPaths.Resolve(Path.Combine("sensors", nameOrPath + ".json"));
    }

    public static SensorConfig Load(string nameOrPath)
    {
        string path = PathFor(nameOrPath);
        SensorConfig c = JsonConvert.DeserializeObject<SensorConfig>(File.ReadAllText(path), Settings);
        c.Resolve();
        return c;
    }

    public static SensorConfig Defaults()
    {
        var c = new SensorConfig();
        c.Resolve();
        return c;
    }

    // what the run logger stores as the resolved copy of the configuration
    public static string ToJson(SensorConfig c) { return JsonConvert.SerializeObject(c, Settings); }
}