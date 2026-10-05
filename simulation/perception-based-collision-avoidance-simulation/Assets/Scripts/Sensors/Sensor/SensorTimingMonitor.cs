using System.Collections.Generic;
using System.Text;
using UnityEngine;

public class SensorTimingMonitor : MonoBehaviour
{
    public LidarSensor lidar;
    public CameraSensor cameraSensor;
    public PoseEstimator pose;
    public float reportEvery = 10f;    // s of simulation time, 0 = only from the context menu

    class Stream
    {
        public string name;
        public readonly List<double> stamps = new List<double>();
        public int inversions, poseMismatches;
        public Stream(string n) { name = n; }
    }

    readonly Stream sLidar = new Stream("lidar");
    readonly Stream sCamera = new Stream("camera");
    readonly Stream sPose = new Stream("pose");
    double maxSeen = double.NegativeInfinity;
    double nextReport, lastSim, lastWall;
    float maxFrameMs;

    void OnEnable()
    {
        if (lidar != null) lidar.ScanReady += OnScan;
        if (cameraSensor != null) cameraSensor.FrameReady += OnFrame;
        if (pose != null) pose.PoseReady += OnPose;
    }

    void OnDisable()
    {
        if (lidar != null) lidar.ScanReady -= OnScan;
        if (cameraSensor != null) cameraSensor.FrameReady -= OnFrame;
        if (pose != null) pose.PoseReady -= OnPose;
    }

    void Start()
    {
        nextReport = reportEvery;
        lastSim = Time.fixedTimeAsDouble;
        lastWall = Time.realtimeSinceStartupAsDouble;
    }

    void Record(Stream s, double stamp)
    {
        s.stamps.Add(stamp);
        if (stamp < maxSeen - 1e-9) s.inversions++;   // arrived after a message with a later capture time
        if (stamp > maxSeen) maxSeen = stamp;
    }

    void OnScan(LidarScan sc)
    {
        Record(sLidar, sc.timestamp);
        if (System.Math.Abs(sc.pose.timestamp - sc.timestamp) > 1e-9) sLidar.poseMismatches++;
    }

    void OnFrame(CameraFrame f)
    {
        Record(sCamera, f.timestamp);
        if (System.Math.Abs(f.pose.timestamp - f.timestamp) > 1e-9) sCamera.poseMismatches++;
    }

    void OnPose(PoseState p) { Record(sPose, p.timestamp); }

    void Update() { maxFrameMs = Mathf.Max(maxFrameMs, Time.unscaledDeltaTime * 1000f); }

    void FixedUpdate()
    {
        if (reportEvery > 0f && Time.fixedTimeAsDouble >= nextReport)
        {
            nextReport += reportEvery;
            Report();
        }
    }

    string Describe(Stream s, double dt)
    {
        var hist = new SortedDictionary<int, int>();
        int off = 0;
        for (int i = 0; i < s.stamps.Count; i++)
        {
            double k = s.stamps[i] / dt;
            if (System.Math.Abs(k - System.Math.Round(k)) > 1e-3) off++;
            if (i >= 2)   // the first interval depends on when the first step happens
            {
                int ms = (int)System.Math.Round((s.stamps[i] - s.stamps[i - 1]) * 1000.0);
                int c;
                hist.TryGetValue(ms, out c);
                hist[ms] = c + 1;
            }
        }
        int total = 0;
        foreach (var kv in hist) total += kv.Value;
        var sb = new StringBuilder(s.name + ": " + s.stamps.Count + " samples | intervals ");
        foreach (var kv in hist) sb.Append(kv.Key + " ms: " + kv.Value + " (" + (100.0 * kv.Value / System.Math.Max(1, total)).ToString("F1") + "%)  ");
        sb.Append("| off-grid stamps " + off + " | arrived out of timestamp order " + s.inversions +
                  " (" + (100.0 * s.inversions / System.Math.Max(1, s.stamps.Count)).ToString("F0") + "%)");
        if (s.name != "pose") sb.Append(" | pose stamp mismatches " + s.poseMismatches);
        return sb.ToString();
    }

    [ContextMenu("Report now")]
    void Report()
    {
        double dt = Time.fixedDeltaTime;
        var sb = new StringBuilder("SENSOR TIMING at t=" + Time.fixedTimeAsDouble.ToString("F2") + " s\n");
        sb.AppendLine("  " + Describe(sLidar, dt));
        sb.AppendLine("  " + Describe(sCamera, dt));
        sb.AppendLine("  " + Describe(sPose, dt));

        // LiDAR scans against camera frames (only scans old enough to have their camera frame delivered)
        List<double> cam = sCamera.stamps;
        if (cam.Count > 0)
        {
            double maxOff = 0.0;
            int same = 0, n = 0;
            foreach (double t in sLidar.stamps)
            {
                if (t > cam[cam.Count - 1]) break;
                int idx = cam.BinarySearch(t);
                double off;
                if (idx >= 0) off = 0.0;
                else
                {
                    idx = ~idx;
                    off = double.MaxValue;
                    if (idx < cam.Count) off = System.Math.Min(off, cam[idx] - t);
                    if (idx > 0) off = System.Math.Min(off, t - cam[idx - 1]);
                }
                n++;
                if (off < 1e-6) same++;
                maxOff = System.Math.Max(maxOff, off);
            }
            sb.AppendLine("  LiDAR scans with a camera frame at the same stamp: " + same + " of " + n + ", largest offset to the nearest frame " + (maxOff * 1000.0).ToString("F0") + " ms");
        }

        double simNow = Time.fixedTimeAsDouble, wallNow = Time.realtimeSinceStartupAsDouble;
        double rtf = (simNow - lastSim) / System.Math.Max(1e-9, wallNow - lastWall);
        sb.AppendLine("  real-time factor since the last report " + rtf.ToString("F3") + " | worst frame " + maxFrameMs.ToString("F1") +
                      " ms | time scale " + Time.timeScale.ToString("F2"));
        lastSim = simNow; lastWall = wallNow; maxFrameMs = 0f;
        UnityEngine.Debug.Log(sb.ToString());
    }

    [ContextMenu("Reset statistics")]
    void ResetStats()
    {
        sLidar.stamps.Clear(); sCamera.stamps.Clear(); sPose.stamps.Clear();
        sLidar.inversions = sCamera.inversions = sPose.inversions = 0;
        sLidar.poseMismatches = sCamera.poseMismatches = 0;
        maxSeen = double.NegativeInfinity;
    }
}