using System;
using System.IO;
using UnityEngine;
using static System.Net.Mime.MediaTypeNames;

public static class ConfigPaths
{
    public static string ConfigDir()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++)
            if (args[i] == "--config-dir") return args[i + 1];

        var dir = new DirectoryInfo(UnityEngine.Application.dataPath);
        while (dir != null)
        {
            string cand = Path.Combine(dir.FullName, "configs");
            if (Directory.Exists(cand)) return cand;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("No 'configs' folder found above " + UnityEngine.Application.dataPath);
    }

    public static string Resolve(string relative)
    {
        return Path.IsPathRooted(relative) ? relative : Path.Combine(ConfigDir(), relative);
    }
}