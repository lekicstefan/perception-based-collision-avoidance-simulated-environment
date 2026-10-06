using UnityEngine;

// Manual check of the LiDAR against a flat wall at a known place. Uses ground-truth hit ids: for checks only.
public class LidarWallCheck : MonoBehaviour
{
    public LidarSensor lidar;
    public BoxCollider wall;            // thin box; its front face (the local -z side) faces the sensor
    public int reportEveryScans = 100;

    long scans, geo, returned, none;
    double sumErr, sumSq, maxAbs, sumExpected, minT = 1e9, maxT;
    bool[] rowHit, colHit;

    void OnEnable() { if (lidar != null) lidar.ScanReady += OnScan; }
    void OnDisable() { if (lidar != null) lidar.ScanReady -= OnScan; }

    void OnScan(LidarScan scan)
    {
        if (wall == null) return;
        if (rowHit == null) { rowHit = new bool[scan.rows]; colHit = new bool[scan.cols]; }
        LidarConfig lc = lidar.rig.Config.lidar;
        int wallId = wall.GetInstanceID();
        Vector3 o = lidar.LastOrigin;
        Vector3 n = -wall.transform.forward;                                   // towards the sensor
        Vector3 p0 = wall.transform.TransformPoint(wall.center - new Vector3(0f, 0f, 0.5f * wall.size.z));
        float num = Vector3.Dot(p0 - o, n);
        double beta = SimEnvironment.FogExtinction * lc.fogSensitivity;
        int[] ids = lidar.GroundTruthHitIds;

        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] != wallId) continue;
            double t = num / Vector3.Dot(lidar.CellDirectionWorld(i), n);       // exact distance to the wall plane
            geo++;
            sumExpected += (1.0 - lc.dropoutProbability) * System.Math.Exp(-beta * t);   // grazing dropout assumed off
            minT = System.Math.Min(minT, t);
            maxT = System.Math.Max(maxT, t);
            rowHit[i / scan.cols] = true;
            colHit[i % scan.cols] = true;

            ushort r = scan.ranges[i];
            if (r == LidarScan.NoReturn) { none++; continue; }
            returned++;
            double e = r * 0.01 - t;
            sumErr += e;
            sumSq += e * e;
            maxAbs = System.Math.Max(maxAbs, System.Math.Abs(e));
        }
        scans++;
        if (reportEveryScans > 0 && scans % reportEveryScans == 0) Report();
    }

    [ContextMenu("Report now")]
    void Report()
    {
        if (geo == 0) { UnityEngine.Debug.Log("LIDAR WALL CHECK: no hits on the wall yet"); return; }
        LidarConfig lc = lidar.rig.Config.lidar;
        int rows = 0, cols = 0, topRow = -1, lowRow = -1, firstCol = -1, lastCol = -1;
        for (int r = 0; r < rowHit.Length; r++) if (rowHit[r]) { rows++; if (topRow < 0) topRow = r; lowRow = r; }
        for (int c = 0; c < colHit.Length; c++) if (colHit[c]) { cols++; if (firstCol < 0) firstCol = c; lastCol = c; }
        double mean = returned > 0 ? sumErr / returned : 0.0;
        double std = returned > 1 ? System.Math.Sqrt(System.Math.Max(0.0, sumSq / returned - mean * mean)) : 0.0;
        UnityEngine.Debug.Log("LIDAR WALL CHECK over " + scans + " scans\n" +
            "  cells hitting the wall per scan " + (geo / (double)scans).ToString("F0") + ", range " + minT.ToString("F2") + " to " + maxT.ToString("F2") + " m\n" +
            "  returned fraction " + (returned / (double)geo).ToString("F4") + " (expected " + (sumExpected / geo).ToString("F4") + "), no return " + none + "\n" +
            "  range error: mean " + (mean * 1000.0).ToString("F2") + " mm, std " + (std * 1000.0).ToString("F2") + " mm, max |e| " + (maxAbs * 1000.0).ToString("F2") + " mm\n" +
            "  field of view: " + rows + " rows hit (top " + lc.ElevationsDeg[topRow].ToString("F2") + " deg, lowest " + lc.ElevationsDeg[lowRow].ToString("F2") +
            " deg), " + cols + " columns hit (azimuth " + lc.AzimuthDeg(firstCol).ToString("F1") + " to " + lc.AzimuthDeg(lastCol).ToString("F1") + " deg)");
    }

    [ContextMenu("Reset statistics")]
    void ResetStats()
    {
        scans = geo = returned = none = 0;
        sumErr = sumSq = maxAbs = sumExpected = 0.0;
        minT = 1e9; maxT = 0.0;
        rowHit = null; colHit = null;
    }
}