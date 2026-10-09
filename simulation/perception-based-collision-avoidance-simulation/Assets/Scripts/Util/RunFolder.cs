using System.IO;
using UnityEngine;

// The folder every log of a run goes into (docs/run_folder.md). The processor decides whether there is one and what it
// is called (python -m server.run_processor --run-dir [name]) and sends the absolute path in the READY message.
//   Unknown  : READY has not arrived yet; loggers wait and write nothing.
//   Disabled : the processor was started without --run-dir (or the network layer is off); loggers switch themselves off.
//   Enabled  : Dir is the run folder (created here if it does not exist yet).
public static class RunFolder
{
    public enum State { Unknown, Disabled, Enabled }

    static string dir;

    public static State Current { get; private set; }
    public static string RunId { get; private set; }
    public static string Dir { get { return dir; } }
    public static string UnityDir { get { return Sub("unity"); } }
    public static string ConfigCopyDir { get { return Sub("config"); } }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetForNewPlaySession() { dir = null; RunId = null; Current = State.Unknown; }   // domain reload may be off

    // Called once by NetworkHost: with the path from READY, or null when there is no run folder.
    public static void Configure(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            Current = State.Disabled;
            return;
        }
        Directory.CreateDirectory(path);
        dir = path;
        RunId = System.IO.Path.GetFileName(path.TrimEnd('/', '\\'));
        Current = State.Enabled;
    }

    static string Sub(string name)
    {
        string p = System.IO.Path.Combine(dir, name);
        Directory.CreateDirectory(p);
        return p;
    }
}