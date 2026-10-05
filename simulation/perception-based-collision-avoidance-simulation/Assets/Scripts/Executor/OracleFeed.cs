using System.Collections.Generic;
using UnityEngine;

// ORACLE MESSAGE TYPES: they exist only in oracle mode and break the information boundary on purpose.
// Results obtained with the oracle are an upper bound, never "the system".
public class OracleObject
{
    public int id;
    public double x, y, z;        // centre, true start frame (x forward, y left, z up)
    public double yaw;            // rad
    public double vx, vy;         // m/s
    public float length, width, height;
}

public class OracleFrame
{
    public double timestamp;      // the LiDAR scan time it belongs to
    public OracleObject[] objects;
}

public class OracleFeed : MonoBehaviour
{
    public bool oracleMode = false;     // off by default. Use with pose noise OFF: the oracle ignores it.
    public LidarSensor lidar;
    public PoseEstimator pose;

    public event System.Action<OracleFrame> OracleReady;
    public OracleFrame Last { get; private set; }

    void OnEnable() { if (lidar != null) lidar.ScanReady += OnScan; }
    void OnDisable() { if (lidar != null) lidar.ScanReady -= OnScan; }

    void Start()
    {
        if (oracleMode)
            UnityEngine.Debug.LogWarning("ORACLE MODE ON: the processor is given ground truth. Results are an upper bound, not the system.");
    }

    void OnScan(LidarScan scan)
    {
        if (!oracleMode || !pose.HasOrigin) return;
        Transform mount = lidar.rig.LidarMount;
        LidarConfig lc = lidar.rig.Config.lidar;
        var list = new List<OracleObject>();
        foreach (Hazard h in Hazard.All)
        {
            Vector3 local = mount.InverseTransformPoint(h.WorldCenter);
            float az = Mathf.Atan2(-local.x, local.z) * Mathf.Rad2Deg;
            if (local.z <= 0f || Mathf.Abs(az) > 0.5f * lc.horizontalFovDeg || local.magnitude > lc.maxRangeM) continue;
            double x, y, z, vx, vy;
            pose.WorldToStart(h.WorldCenter, out x, out y, out z);
            pose.VelocityToStart(h.WorldVelocity, out vx, out vy);
            list.Add(new OracleObject
            {
                id = h.Id,
                x = x,
                y = y,
                z = z,
                yaw = pose.YawToStart(h.transform.eulerAngles.y),
                vx = vx,
                vy = vy,
                length = h.Length,
                width = h.Width,
                height = h.Height
            });
        }
        Last = new OracleFrame { timestamp = scan.timestamp, objects = list.ToArray() };

        if (scan.frameId < 3 || scan.frameId % 50 == 0)
        {
            var s = new System.Text.StringBuilder("ORACLE t=" + scan.timestamp.ToString("F3") + " | " + Last.objects.Length + " objects:");
            foreach (OracleObject o in Last.objects)
                s.Append(" #" + o.id + " (" + o.x.ToString("F2") + ", " + o.y.ToString("F2") + ", " + o.z.ToString("F2") + ") v(" +
                         o.vx.ToString("F2") + ", " + o.vy.ToString("F2") + ") yaw " + (o.yaw * 180.0 / System.Math.PI).ToString("F1") + " |");
            UnityEngine.Debug.Log(s.ToString());
        }
        if (OracleReady != null) OracleReady(Last);
    }
}