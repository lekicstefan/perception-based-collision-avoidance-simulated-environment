using System.Collections.Generic;

public static class Cli
{
    public static string Get(string key, string defaultValue)
    {
        string[] a = System.Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < a.Length; i++) if (a[i] == key) return a[i + 1];
        return defaultValue;
    }

    public static int GetInt(string key, int defaultValue)
    {
        int v;
        return int.TryParse(Get(key, null), out v) ? v : defaultValue;
    }

    // every value of an argument that may be repeated, in order: --set a=1 --set b=2
    public static List<string> GetAll(string key)
    {
        var values = new List<string>();
        string[] a = System.Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < a.Length; i++) if (a[i] == key) values.Add(a[i + 1]);
        return values;
    }
}