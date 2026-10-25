using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;

// Time stamps for the communication test. All times are Stopwatch ticks converted to ns: on Windows that is the performance
// counter, the same clock as Python's time.perf_counter_ns(), so the two latency.csv files can be merged by the capture time in ms.
// 0 in the CSV means the event did not happen (a command that was replaced by a newer one before it was applied).
public static class Latency
{
    struct Row
    {
        public double simTime;
        public long capture, lidarReady, cameraReady, built, sent, cmdReceived, cmdApplied;
        public int bytes;
    }

    static readonly Dictionary<uint, Row> rows = new Dictionary<uint, Row>();
    static readonly object gate = new object();
    static readonly double nsPerTick = 1e9 / Stopwatch.Frequency;

    public static uint Key(double simTime) { return (uint)System.Math.Round(simTime * 1000.0); }   // capture time in ms identifies a frame
    public static long Ns() { return TicksToNs(Stopwatch.GetTimestamp()); }
    public static long TicksToNs(long ticks) { return (long)(ticks * nsPerTick); }

    public static void Reset() { lock (gate) rows.Clear(); }

    public static void Frame(uint id, double simTime, long capture, long lidarReady, long cameraReady, long built, int bytes)
    {
        lock (gate)
            rows[id] = new Row { simTime = simTime, capture = capture, lidarReady = lidarReady, cameraReady = cameraReady, built = built, bytes = bytes };
    }

    public static void Sent(uint id, long ns) { Update(id, 0, ns); }
    public static void CommandReceived(uint id, long ns) { Update(id, 1, ns); }
    public static void CommandApplied(uint id, long ns) { Update(id, 2, ns); }

    static void Update(uint id, int what, long ns)
    {
        lock (gate)
        {
            Row r;
            if (!rows.TryGetValue(id, out r)) return;
            if (what == 0) r.sent = ns;
            else if (what == 1) r.cmdReceived = ns;
            else r.cmdApplied = ns;
            rows[id] = r;
        }
    }

    public static void Write(string path)
    {
        var ids = new List<uint>();
        var sb = new StringBuilder("t_ms,sim_t,capture_ns,lidar_ready_ns,camera_ready_ns,built_ns,sent_ns,cmd_received_ns,cmd_applied_ns,bytes\n");
        lock (gate)
        {
            ids.AddRange(rows.Keys);
            ids.Sort();
            foreach (uint id in ids)
            {
                Row r = rows[id];
                sb.Append(id).Append(',').Append(r.simTime.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                  .Append(r.capture).Append(',').Append(r.lidarReady).Append(',').Append(r.cameraReady).Append(',')
                  .Append(r.built).Append(',').Append(r.sent).Append(',').Append(r.cmdReceived).Append(',')
                  .Append(r.cmdApplied).Append(',').Append(r.bytes).Append('\n');
            }
        }
        File.WriteAllText(path, sb.ToString());
        UnityEngine.Debug.Log("Latency: " + ids.Count + " frames written to " + path);
    }
}