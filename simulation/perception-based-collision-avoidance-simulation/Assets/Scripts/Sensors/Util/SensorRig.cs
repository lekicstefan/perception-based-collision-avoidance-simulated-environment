using UnityEngine;

[DefaultExecutionOrder(-200)] // config ready before everything else wakes up
public class SensorRig : MonoBehaviour
{
    public Vehicle vehicle;
    public string configName = "default";
    public int seedOverride = -1;          // >= 0 overrides the seed in the config; --seed N overrides both

    public SensorConfig Config { get; private set; }
    public int Seed { get; private set; }
    public Transform LidarMount { get; private set; }
    public Transform CameraMount { get; private set; }

    public int DerivedSeed(string salt) { return SeedUtil.Derive(Seed, salt); }

    void Awake()
    {
        string cfgName = Cli.Get("--sensors", configName);
        Config = SensorConfigLoader.Load(cfgName);
        int cliSeed = Cli.GetInt("--seed", -1);
        Seed = cliSeed >= 0 ? cliSeed : (seedOverride >= 0 ? seedOverride : Config.seed);

        LidarMount = CreateMount("LidarMount", Config.lidar.mount);
        CameraMount = CreateMount("CameraMount", Config.camera.mount);

        float dt = Time.fixedDeltaTime;
        string note = "";
        CheckRate("camera", Config.camera.rateHz, dt, ref note);
        CheckRate("lidar", Config.lidar.rateHz, dt, ref note);
        CheckRate("pose", Config.pose.rateHz, dt, ref note);
        UnityEngine.Debug.Log("SENSOR RIG '" + cfgName + "' | seed " + Seed + " | camera " + Config.camera.width + "x" +
            Config.camera.height + " @ " + Config.camera.rateHz.ToString("F0") + " Hz | lidar " + Config.lidar.Rows + " x " +
            Config.lidar.Columns + " @ " + Config.lidar.rateHz.ToString("F0") + " Hz | pose @ " + Config.pose.rateHz.ToString("F0") + " Hz" + note);
    }

    static void CheckRate(string label, float rateHz, float dt, ref string note)
    {
        float steps = (1f / rateHz) / dt;
        if (steps < 1f - 1e-3f) throw new System.ArgumentException("Sensor config: " + label + " rate is faster than the physics step");
        if (Mathf.Abs(steps - Mathf.Round(steps)) > 1e-3f)
            note += " | " + label + ": " + steps.ToString("F2") + " steps per sample, captured at the nearest fixed step";
    }

    Transform CreateMount(string objectName, Mount m)
    {
        var go = new GameObject(objectName);
        go.transform.SetParent(vehicle.transform, false);
        go.transform.localPosition = Frames.ToUnity(m.x, m.y, m.z);
        go.transform.localRotation = Frames.MountRotationToUnity(m.yawDeg, m.pitchDeg, m.rollDeg);
        return go.transform;
    }

    void OnDrawGizmos()
    {
        if (!UnityEngine.Application.isPlaying || Config == null || LidarMount == null) return;
        Matrix4x4 old = Gizmos.matrix;

        Gizmos.matrix = LidarMount.localToWorldMatrix;
        Gizmos.color = Color.yellow;
        float half = Config.lidar.horizontalFovDeg * 0.5f * Mathf.Deg2Rad;
        Vector3 left = new Vector3(-Mathf.Sin(half), 0f, Mathf.Cos(half)) * 15f;
        Vector3 right = new Vector3(Mathf.Sin(half), 0f, Mathf.Cos(half)) * 15f;
        Gizmos.DrawLine(Vector3.zero, left);
        Gizmos.DrawLine(Vector3.zero, right);
        Gizmos.DrawLine(left, right);
        Gizmos.DrawSphere(Vector3.zero, 0.1f);

        Gizmos.matrix = CameraMount.localToWorldMatrix;
        Gizmos.color = Color.cyan;
        float aspect = (float)Config.camera.width / Config.camera.height;
        Gizmos.DrawFrustum(Vector3.zero, Config.camera.VerticalFovDeg, 15f, 0.3f, aspect);

        Gizmos.matrix = old;
    }
}