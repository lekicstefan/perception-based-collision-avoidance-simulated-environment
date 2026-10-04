using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

[DefaultExecutionOrder(20)] // after Vehicle and the scripted objects of the same step
public class LidarSensor : MonoBehaviour
{
    public SensorRig rig;
    public LayerMask excludeLayers;     // layers the rays ignore: the ego car (Ego)
    public bool drawPointCloud = true;  // Scene view while playing
    public int gizmoStride = 1;         // draw every n-th cell
    public bool showStats = true;       // on-screen line
    public int logEveryScans = 50;

    public event System.Action<LidarScan> ScanReady;
    public LidarScan LastScan { get; private set; }
    public PoseEstimator poseEstimator;

    LidarConfig cfg;
    int rows, cols, n, frameId;
    Vector3[] localDirs;
    NativeArray<RaycastCommand> commands;
    NativeArray<RaycastHit> results;
    QueryParameters query;
    System.Random rng;
    SensorSchedule schedule;

    // debug only, never sent to the processor
    byte[] debugClass;   // 0 none/sky, 1 ground-like surface, 2 other surface
    Vector3 lastOrigin;
    Quaternion lastRot;
    Vector3[] bufGround, bufOther;
    int lastValid, lastNone, lastSky, lastGround, lastOther;
    float lastMs;
    uint lastChecksum;
    float gMin, gMax, oMin, oMax;

    void Start()
    {
        cfg = rig.Config.lidar;
        rows = cfg.Rows;
        cols = cfg.Columns;
        n = rows * cols;

        // direction table in the mount frame: processor frame (cos e cos a, cos e sin a, sin e) -> Unity
        localDirs = new Vector3[n];
        for (int r = 0; r < rows; r++)
        {
            float e = cfg.ElevationsDeg[r] * Mathf.Deg2Rad;
            for (int c = 0; c < cols; c++)
            {
                float a = cfg.AzimuthDeg(c) * Mathf.Deg2Rad;
                localDirs[r * cols + c] = Frames.ToUnity(Mathf.Cos(e) * Mathf.Cos(a), Mathf.Cos(e) * Mathf.Sin(a), Mathf.Sin(e));
            }
        }

        commands = new NativeArray<RaycastCommand>(n, Allocator.Persistent);
        results = new NativeArray<RaycastHit>(n, Allocator.Persistent);
        query = new QueryParameters(~excludeLayers.value, false, QueryTriggerInteraction.Ignore, false);
        rng = new System.Random(rig.DerivedSeed("lidar"));
        schedule = new SensorSchedule(cfg.rateHz);
        debugClass = new byte[n];
        bufGround = new Vector3[2 * n];
        bufOther = new Vector3[2 * n];
        if (excludeLayers.value == 0)
            UnityEngine.Debug.LogWarning("LidarSensor: no layers excluded, the rays will hit the ego car's colliders");
    }

    void FixedUpdate()
    {
        if (!cfg.enabled) return;
        double now = Time.fixedTimeAsDouble;
        if (!schedule.Due(now, Time.fixedDeltaTime)) return;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        Physics.SyncTransforms(); // objects moved by scripts in this step must be current

        Transform mount = rig.LidarMount;
        lastOrigin = mount.position;
        lastRot = mount.rotation;

        for (int i = 0; i < n; i++)
            commands[i] = new RaycastCommand(lastOrigin, lastRot * localDirs[i], query, cfg.maxRangeM);
        JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, 64, 1, default(JobHandle));
        handle.Complete();

        var scan = new LidarScan
        {
            timestamp = now,
            frameId = frameId++,
            rows = rows,
            cols = cols,
            ranges = new ushort[n],
            pose = poseEstimator != null ? poseEstimator.Current : default(PoseState)
        };

        Process(scan.ranges);
        LastScan = scan;
        lastChecksum = Checksum(scan.ranges);
        lastMs = (float)sw.Elapsed.TotalMilliseconds;

        if (scan.frameId < 3 || (logEveryScans > 0 && scan.frameId % logEveryScans == 0))
            UnityEngine.Debug.Log("LIDAR scan " + scan.frameId + " t=" + now.ToString("F3") + " | valid " + lastValid +
                " (ground-like " + lastGround + " [" + gMin.ToString("F2") + ", " + gMax.ToString("F2") + "] m, other " + lastOther +
                " [" + oMin.ToString("F2") + ", " + oMax.ToString("F2") + "] m) | no return " + lastNone + " | sky/max range " + lastSky +
                " | checksum " + lastChecksum.ToString("X8") + " | compute " + lastMs.ToString("F2") + " ms" +
                " | pose stamp " + scan.pose.timestamp.ToString("F3"));

        if (ScanReady != null) ScanReady(scan);
    }

    void Process(ushort[] img)
    {
        float maxR = cfg.maxRangeM;
        float beta = SimEnvironment.FogExtinction * cfg.fogSensitivity;   // 1/m
        lastValid = lastNone = lastSky = lastGround = lastOther = 0;
        gMin = oMin = 1e9f; gMax = oMax = 0f;

        for (int i = 0; i < n; i++)
        {
            // the same five random numbers for every cell, whatever happens to it
            double uFog = rng.NextDouble(), uDrop = rng.NextDouble(), uGraze = rng.NextDouble();
            double u1 = rng.NextDouble(), u2 = rng.NextDouble();

            RaycastHit hit = results[i];
            debugClass[i] = 0;
            if (hit.colliderEntityId == 0)   // nothing within range
            {
                img[i] = LidarScan.MaxRangeOrSky;
                lastSky++;
                continue;
            }

            float d = hit.distance;
            bool drop = false;
            if (beta > 0f && uFog > System.Math.Exp(-beta * d)) drop = true;
            if (!drop && uDrop < cfg.dropoutProbability) drop = true;
            if (!drop && cfg.grazingDropoutEnabled)
            {
                float cosInc = Mathf.Abs(Vector3.Dot(commands[i].direction, hit.normal));
                float incDeg = Mathf.Acos(Mathf.Clamp01(cosInc)) * Mathf.Rad2Deg;
                float p = Mathf.Clamp01((incDeg - cfg.grazingStartDeg) / (cfg.grazingEndDeg - cfg.grazingStartDeg));
                if (uGraze < p) drop = true;
            }
            if (drop)
            {
                img[i] = LidarScan.NoReturn;
                lastNone++;
                continue;
            }

            float noise = 0f;
            if (cfg.rangeNoiseStdM > 0f)
                noise = (float)(cfg.rangeNoiseStdM * System.Math.Sqrt(-2.0 * System.Math.Log(System.Math.Max(u1, 1e-12))) *
                                System.Math.Cos(2.0 * System.Math.PI * u2));
            float dn = Mathf.Clamp(d + noise, 0.01f, maxR);
            img[i] = (ushort)Mathf.Clamp(Mathf.RoundToInt(dn * 100f), 1, 65534);
            lastValid++;

            if (hit.normal.y > 0.85f)
            {
                debugClass[i] = 1; lastGround++;
                gMin = Mathf.Min(gMin, d); gMax = Mathf.Max(gMax, d);
            }
            else
            {
                debugClass[i] = 2; lastOther++;
                oMin = Mathf.Min(oMin, d); oMax = Mathf.Max(oMax, d);
            }
        }
    }

    static uint Checksum(ushort[] a)
    {
        unchecked
        {
            uint h = 2166136261u;
            for (int i = 0; i < a.Length; i++)
            {
                h ^= (uint)(a[i] & 0xFF); h *= 16777619u;
                h ^= (uint)(a[i] >> 8); h *= 16777619u;
            }
            return h;
        }
    }

    void OnGUI()
    {
        if (!showStats || LastScan == null) return;
        GUI.Label(new Rect(10, Screen.height - 28, 900, 22),
            "LiDAR scan " + LastScan.frameId + " | valid " + lastValid + " | no return " + lastNone + " | sky " + lastSky +
            " | compute " + lastMs.ToString("F2") + " ms");
    }

    void OnDrawGizmos()
    {
        if (!drawPointCloud || LastScan == null) return;
        int stride = Mathf.Max(1, gizmoStride);
        int ng = 0, no = 0;
        ushort[] img = LastScan.ranges;
        for (int i = 0; i < n; i += stride)
        {
            ushort v = img[i];
            if (v == LidarScan.NoReturn || v == LidarScan.MaxRangeOrSky) continue;
            Vector3 p = lastOrigin + lastRot * localDirs[i] * (v * 0.01f);
            Vector3 q = p + Vector3.up * 0.15f;
            if (debugClass[i] == 1) { bufGround[ng++] = p; bufGround[ng++] = q; }
            else { bufOther[no++] = p; bufOther[no++] = q; }
        }
        Gizmos.color = Color.green;
        Gizmos.DrawLineList(new System.ReadOnlySpan<Vector3>(bufGround, 0, ng));
        Gizmos.color = Color.red;
        Gizmos.DrawLineList(new System.ReadOnlySpan<Vector3>(bufOther, 0, no));
    }

    void OnDestroy()
    {
        if (commands.IsCreated) commands.Dispose();
        if (results.IsCreated) results.Dispose();
    }
}