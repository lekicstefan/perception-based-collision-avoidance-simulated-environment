using System;
using System.IO;
using UnityEngine;

// The one folder every log of a run goes into (docs/run_folder.md):
//   runs/<run id>/   run.json   config/   unity/   python/
// Created on first use. Command line: --runs-root <folder> and --run-id <name> (the batch runner of phase 14 uses them).
public static class RunFolder
{
    static string dir;

    public static string RunId { get; private set; }
    public static string Dir { get { Ensure(); return dir; } }
    public static string UnityDir { get { return Sub("unity"); } }
    public static string ConfigCopyDir { get { return Sub("config"); } }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetForNewPlaySession() { dir = null; RunId = null; }   // in case domain reload is switched off

    static string Sub(string name)
    {
        string p = System.IO.Path.Combine(Dir, name);
        Directory.CreateDirectory(p);
        return p;
    }

    static void Ensure()
    {
        if (dir != null) return;
        string root = Cli.Get("--runs-root", null);
        if (root == null) root = System.IO.Path.Combine(Directory.GetParent(ConfigPaths.ConfigDir()).FullName, "runs");
        string id = Cli.Get("--run-id", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        string p = System.IO.Path.Combine(root, id);
        int n = 2;
        while (Directory.Exists(p)) p = System.IO.Path.Combine(root, id + "_" + n++);
        Directory.CreateDirectory(p);
        dir = p;
        RunId = System.IO.Path.GetFileName(p);
    }
}