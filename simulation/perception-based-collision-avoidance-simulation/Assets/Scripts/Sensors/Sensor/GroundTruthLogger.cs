using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Writes the ground truth of a run for the evaluation. Never part of the communication path.
[DefaultExecutionOrder(30)] // after the sensors (20, 25) of the same step
public class GroundTruthLogger : MonoBehaviour
{
    public Vehicle vehicle;
    public PoseEstimator pose;
    public LidarSensor lidar;
    public string outputFolder = "testing/groundtruth";   // relative to the repository root
    public string runName = "";           // empty = date and time
    public int logEveryNSteps = 1;        // 1 = every physics step
    public bool useRunFolder = true;
    public bool persistData = false;

    StreamWriter ego, haz, vis, info;
    readonly HashSet<int> infoWritten = new HashSet<int>();
    long step;
    double nextFlush;
    bool closed;
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    static string F(double v) { return v.ToString("F4", Inv); }

    void Start()
    {
        if (!persistData) return;

        string folder;
        if (useRunFolder) folder = RunFolder.UnityDir;
        else
        {
            string root = Directory.GetParent(ConfigPaths.ConfigDir()).FullName;
            string name = string.IsNullOrEmpty(runName) ? System.DateTime.Now.ToString("yyyyMMdd_HHmmss") : runName;
            folder = Path.Combine(root, outputFolder, name);
        }
        Directory.CreateDirectory(folder);

        ego = new StreamWriter(Path.Combine(folder, "ego.csv"));
        ego.WriteLine("t,x,y,z,yaw_rad,pitch_rad,speed,yaw_rate,steering,accel,jerk,requested_accel");
        haz = new StreamWriter(Path.Combine(folder, "hazards.csv"));
        haz.WriteLine("t,id,x,y,z,yaw_rad,vx,vy,speed,length,width,height");
        vis = new StreamWriter(Path.Combine(folder, "visibility.csv"));
        vis.WriteLine("t,scan,id,hits_geometric,hits_returned,min_range_m,centre_range_m,azimuth_deg,in_view");
        info = new StreamWriter(Path.Combine(folder, "hazard_info.csv"));
        info.WriteLine("id,name,label,length,width,height,colliders");
        lidar.ScanReady += OnScan;
        nextFlush = Time.realtimeSinceStartupAsDouble + 2.0;
        UnityEngine.Debug.Log("GROUND TRUTH logging to " + folder);
    }

    void FixedUpdate()
    {
        if (closed || !pose.HasOrigin) return;
        step++;
        if (step % Mathf.Max(1, logEveryNSteps) != 0) return;

        PoseState p = pose.GroundTruthPose;
        ego.WriteLine(F(p.timestamp) + "," + F(p.x) + "," + F(p.y) + "," + F(p.z) + "," + F(p.yaw) + "," + F(p.pitch) + "," +
            F(p.speed) + "," + F(p.yawRate) + "," + F(p.steeringAngle) + "," + F(vehicle.Accel) + "," + F(vehicle.Jerk) + "," + F(vehicle.RequestedAccel));

        foreach (Hazard h in Hazard.All)
        {
            if (infoWritten.Add(h.Id))
                info.WriteLine(h.Id + "," + h.name + "," + h.label + "," + F(h.Length) + "," + F(h.Width) + "," + F(h.Height) + "," + h.ColliderIds.Count);
            double x, y, z, vx, vy;
            pose.WorldToStart(h.WorldCenter, out x, out y, out z);
            pose.VelocityToStart(h.WorldVelocity, out vx, out vy);
            double yaw = pose.YawToStart(h.transform.eulerAngles.y);
            haz.WriteLine(F(p.timestamp) + "," + h.Id + "," + F(x) + "," + F(y) + "," + F(z) + "," + F(yaw) + "," + F(vx) + "," + F(vy) + "," +
                F(System.Math.Sqrt(vx * vx + vy * vy)) + "," + F(h.Length) + "," + F(h.Width) + "," + F(h.Height));
        }

        if (Time.realtimeSinceStartupAsDouble >= nextFlush)
        {
            Flush();
            nextFlush = Time.realtimeSinceStartupAsDouble + 2.0;
        }
    }

    // called at every LiDAR scan: how many cells hit each hazard, and where the hazard is relative to the sensor
    void OnScan(LidarScan scan)
    {
        if (closed || !pose.HasOrigin) return;
        List<Hazard> list = Hazard.All;
        var map = new Dictionary<int, int>();      // collider id -> index in the hazard list
        for (int k = 0; k < list.Count; k++)
            foreach (int cid in list[k].ColliderIds) map[cid] = k;
        var geo = new int[list.Count];
        var ret = new int[list.Count];
        var minR = new float[list.Count];
        for (int k = 0; k < minR.Length; k++) minR[k] = float.MaxValue;

        int[] ids = lidar.GroundTruthHitIds;
        float[] ranges = lidar.GroundTruthHitRanges;
        ushort[] img = scan.ranges;
        for (int i = 0; i < ids.Length; i++)
        {
            int k;
            if (ids[i] == 0 || !map.TryGetValue(ids[i], out k)) continue;
            geo[k]++;
            if (img[i] != LidarScan.NoReturn && img[i] != LidarScan.MaxRangeOrSky) ret[k]++;
            if (ranges[i] < minR[k]) minR[k] = ranges[i];
        }

        Transform mount = lidar.rig.LidarMount;
        LidarConfig lc = lidar.rig.Config.lidar;
        for (int k = 0; k < list.Count; k++)
        {
            Vector3 local = mount.InverseTransformPoint(list[k].WorldCenter);   // Unity: x right, z forward
            float az = Mathf.Atan2(-local.x, local.z) * Mathf.Rad2Deg;          // positive to the left
            float centreRange = local.magnitude;
            bool inView = local.z > 0f && Mathf.Abs(az) <= 0.5f * lc.horizontalFovDeg && centreRange <= lc.maxRangeM;
            vis.WriteLine(F(scan.timestamp) + "," + scan.frameId + "," + list[k].Id + "," + geo[k] + "," + ret[k] + "," +
                (geo[k] > 0 ? F(minR[k]) : "") + "," + F(centreRange) + "," + F(az) + "," + (inView ? 1 : 0));
        }
    }

    public void Flush()
    {
        if (closed) return;
        ego.Flush(); haz.Flush(); vis.Flush(); info.Flush();
    }

    void Close()
    {
        if (closed) return;
        closed = true;
        ego.Dispose(); haz.Dispose(); vis.Dispose(); info.Dispose();
    }

    void OnApplicationQuit() { Close(); }
    void OnDestroy() { Close(); }
}