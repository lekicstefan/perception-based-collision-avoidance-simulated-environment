using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

// Development recorder (step 3.9): writes the sensor outputs and the ground truth of a run to disk so the
// processor can be developed and tested offline (phases 5 to 7). It is NOT the replay format of step 4.8.
[DefaultExecutionOrder(40)]
public class DevRecorder : MonoBehaviour
{
    public SensorRig rig;
    public LidarSensor lidar;
    public CameraSensor cameraSensor;
    public PoseEstimator pose;
    public CalibrationProvider calibration;
    public GroundTruthLogger groundTruth;
    public string outputFolder = "testing/data/recordings"; // relative to the repository root
    public string sceneName = "dev";
    public float stopAfterSeconds = 21f;    // simulation time; 0 = never
    public bool persistData = false;

    string folder;
    bool started, closed, warnedRaw;
    int nScans, nFrames, nPose;
    BinaryWriter lidarWriter;
    StreamWriter camCsv, poseCsv;
    byte[] rangeBytes;
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    static string R(double v) { return v.ToString("R", Inv); }
    static string R(float v) { return v.ToString("R", Inv); }

    void Awake()
    {
        if (!persistData) return;

        string root = Directory.GetParent(ConfigPaths.ConfigDir()).FullName;
        string name = sceneName + "_" + rig.Config.name + "_seed" + rig.Seed;
        string path = Path.Combine(root, outputFolder, name);
        int suffix = 2;
        while (Directory.Exists(path)) path = Path.Combine(root, outputFolder, name + "_" + suffix++);
        folder = path;
        if (groundTruth != null)    // the ground truth goes into the same folder
        {
            groundTruth.useRunFolder = false;
            groundTruth.outputFolder = outputFolder;
            groundTruth.runName = Path.GetFileName(folder);
        }
    }

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

    void EnsureStarted()
    {
        if (started || !persistData) return;
        started = true;
        Directory.CreateDirectory(Path.Combine(folder, "camera"));
        LidarConfig lc = rig.Config.lidar;
        var meta = new
        {
            formatVersion = 1,
            scene = sceneName,
            sensorConfig = rig.Config.name,
            seed = rig.Seed,
            unityVersion = UnityEngine.Application.unityVersion,
            fixedDeltaTime = Time.fixedDeltaTime,
            rows = lc.Rows,
            cols = lc.Columns,
            lidarRateHz = lc.rateHz,
            cameraRateHz = rig.Config.camera.rateHz,
            cameraFormat = rig.Config.camera.format,
            poseRateHz = rig.Config.pose.rateHz,
            recordedAtUtc = System.DateTime.UtcNow.ToString("o"),
            lidarRecord = "int32 frame, float64 t, float64[6] pose (x y z yaw pitch roll), float32[3] (speed yawRate steering), " +
                          "uint16[rows*cols] ranges in cm, row-major, little-endian, packed"
        };
        File.WriteAllText(Path.Combine(folder, "meta.json"), JsonConvert.SerializeObject(meta, Formatting.Indented));
        File.WriteAllText(Path.Combine(folder, "sensors_resolved.json"), SensorConfigLoader.ToJson(rig.Config));
        if (calibration != null && calibration.Json != null)
            File.WriteAllText(Path.Combine(folder, "calibration.json"), calibration.Json);

        lidarWriter = new BinaryWriter(File.Create(Path.Combine(folder, "lidar.bin")));
        camCsv = new StreamWriter(Path.Combine(folder, "camera.csv"));
        camCsv.WriteLine("frame,t,x,y,z,yaw,pitch,roll,speed,yaw_rate,steering,bytes");
        poseCsv = new StreamWriter(Path.Combine(folder, "pose.csv"));
        poseCsv.WriteLine("t,x,y,z,yaw,pitch,roll,speed,yaw_rate,steering");
        UnityEngine.Debug.Log("RECORDING to " + folder);
    }

    static string PoseColumns(PoseState p)
    {
        return R(p.x) + "," + R(p.y) + "," + R(p.z) + "," + R(p.yaw) + "," + R(p.pitch) + "," + R(p.roll) + "," +
               R(p.speed) + "," + R(p.yawRate) + "," + R(p.steeringAngle);
    }

    void OnScan(LidarScan s)
    {
        if (closed || !persistData) return;
        EnsureStarted();
        BinaryWriter w = lidarWriter;
        w.Write(s.frameId);
        w.Write(s.timestamp);
        w.Write(s.pose.x); w.Write(s.pose.y); w.Write(s.pose.z);
        w.Write(s.pose.yaw); w.Write(s.pose.pitch); w.Write(s.pose.roll);
        w.Write(s.pose.speed); w.Write(s.pose.yawRate); w.Write(s.pose.steeringAngle);
        if (rangeBytes == null || rangeBytes.Length != s.ranges.Length * 2) rangeBytes = new byte[s.ranges.Length * 2];
        System.Buffer.BlockCopy(s.ranges, 0, rangeBytes, 0, rangeBytes.Length);
        w.Write(rangeBytes);
        nScans++;
    }

    void OnFrame(CameraFrame f)
    {
        if (closed || !persistData) return;
        EnsureStarted();
        if (f.format != "jpeg")
        {
            if (!warnedRaw) { warnedRaw = true; UnityEngine.Debug.LogWarning("DevRecorder stores JPEG frames only"); }
            return;
        }
        File.WriteAllBytes(Path.Combine(folder, "camera", f.frameId.ToString("D6") + ".jpg"), f.data);
        camCsv.WriteLine(f.frameId + "," + R(f.timestamp) + "," + PoseColumns(f.pose) + "," + f.data.Length);
        nFrames++;
    }

    void OnPose(PoseState p)
    {
        if (closed || !persistData) return;
        EnsureStarted();
        poseCsv.WriteLine(R(p.timestamp) + "," + PoseColumns(p));
        nPose++;
    }

    void FixedUpdate()
    {
        if (closed || stopAfterSeconds <= 0f || Time.fixedTimeAsDouble < stopAfterSeconds) return;
        Close();
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        UnityEngine.Application.Quit();
#endif
    }

    void Close()
    {
        if (closed) return;
        closed = true;
        if (lidarWriter != null) lidarWriter.Dispose();
        if (camCsv != null) camCsv.Dispose();
        if (poseCsv != null) poseCsv.Dispose();
        UnityEngine.Debug.Log("RECORDING finished: " + nScans + " scans, " + nFrames + " camera frames, " + nPose + " pose rows in " + folder);
    }

    void OnApplicationQuit() { Close(); }
    void OnDestroy() { Close(); }
}