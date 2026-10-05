using UnityEngine;

[DefaultExecutionOrder(15)] // after Vehicle (-100), before the sensors (20, 25)
public class PoseEstimator : MonoBehaviour
{
    public SensorRig rig;
    public Vehicle vehicle;
    public bool drawGizmo = true;       // true pose (blue) to reported pose (red), Scene view while playing
    public bool showStats = true;
    public float logInterval = 5f;      // s of simulation time, 0 = off

    public event System.Action<PoseState> PoseReady;      // at the configured pose rate
    public PoseState Current { get; private set; }        // what the processor is told
    public PoseState GroundTruthPose { get; private set; } // EVALUATION ONLY: never put this into a message

    PoseConfig cfg;
    PoseNoiseModel noise;
    SensorSchedule schedule;
    bool haveOrigin;
    double x0, z0, y0, psi0;      // origin: rear axle (Unity x, z), elevation (Unity y) and heading at the start
    double offX, offY, offYaw;
    double nextLog;

    public void ResetOrigin() { haveOrigin = false; }
    public bool HasOrigin { get { return haveOrigin; } }

    public void WorldToStart(Vector3 p, out double x, out double y, out double z)
    {
        double ignoredYaw;
        ToStartFrame(x0, z0, psi0, p.x, p.z, psi0, out x, out y, out ignoredYaw);
        z = p.y - y0;
    }

    public void VelocityToStart(Vector3 v, out double vx, out double vy)
    {
        double c = System.Math.Cos(psi0), s = System.Math.Sin(psi0);
        vx = v.x * c + v.z * s;
        vy = -v.x * s + v.z * c;
    }

    // Unity yaw in degrees -> start-frame yaw in radians
    public double YawToStart(float unityYawDeg)
    {
        double psi = (90.0 - unityYawDeg) * System.Math.PI / 180.0;
        return System.Math.IEEERemainder(psi - psi0, 2.0 * System.Math.PI);
    }

    // Unity ground-plane pose -> start frame (x forward, y left). psi is CCW from +X seen from above.
    public static void ToStartFrame(double x0, double z0, double psi0, double x, double z, double psi,
                                    out double xr, out double yr, out double yaw)
    {
        double dx = x - x0, dz = z - z0;
        double c = System.Math.Cos(psi0), s = System.Math.Sin(psi0);
        xr = dx * c + dz * s;
        yr = -dx * s + dz * c;
        yaw = System.Math.IEEERemainder(psi - psi0, 2.0 * System.Math.PI);
    }

    void Start()
    {
        cfg = rig.Config.pose;
        noise = new PoseNoiseModel(cfg, rig.DerivedSeed("pose"));
        schedule = new SensorSchedule(cfg.rateHz);
        nextLog = logInterval;
    }

    void FixedUpdate()
    {
        double now = Time.fixedTimeAsDouble;
        if (!haveOrigin)
        {
            x0 = vehicle.X; z0 = vehicle.Z; y0 = vehicle.transform.position.y; psi0 = vehicle.Heading;
            haveOrigin = true;
        }

        double xr, yr, yaw;
        ToStartFrame(x0, z0, psi0, vehicle.X, vehicle.Z, vehicle.Heading, out xr, out yr, out yaw);
        var truth = new PoseState
        {
            timestamp = now,
            x = xr,
            y = yr,
            z = vehicle.transform.position.y - y0,
            yaw = yaw,
            pitch = vehicle.PitchDeg * System.Math.PI / 180.0,
            roll = 0.0,
            speed = vehicle.Speed,
            yawRate = vehicle.YawRate,
            steeringAngle = vehicle.SteeringAngle
        };
        GroundTruthPose = truth;

        PoseState reported = truth;
        offX = offY = offYaw = 0.0;
        if (cfg.noiseEnabled)
        {
            noise.Step(Time.fixedDeltaTime, out offX, out offY, out offYaw);
            reported.x += offX;
            reported.y += offY;
            reported.yaw = System.Math.IEEERemainder(yaw + offYaw, 2.0 * System.Math.PI);
        }
        Current = reported;

        if (schedule.Due(now, Time.fixedDeltaTime) && PoseReady != null) PoseReady(reported);

        if (logInterval > 0f && now >= nextLog)
        {
            nextLog += logInterval;
            UnityEngine.Debug.Log("POSE t=" + now.ToString("F2") + " | x " + reported.x.ToString("F2") + " y " + reported.y.ToString("F2") +
                " z " + reported.z.ToString("F2") + " | yaw " + (reported.yaw * 180.0 / System.Math.PI).ToString("F2") + " deg pitch " +
                (reported.pitch * 180.0 / System.Math.PI).ToString("F2") + " deg | speed " + reported.speed.ToString("F2") +
                " yaw rate " + (reported.yawRate * 180f / Mathf.PI).ToString("F2") + " deg/s steering " +
                (reported.steeringAngle * 180f / Mathf.PI).ToString("F2") + " deg" +
                (cfg.noiseEnabled ? " | noise on, offset " + offX.ToString("F3") + ", " + offY.ToString("F3") + " m" : " | noise off"));
        }
    }

    void OnGUI()
    {
        if (!showStats) return;
        PoseState p = Current;
        GUI.Label(new Rect(10, Screen.height - 50, 900, 22),
            "pose x " + p.x.ToString("F2") + " y " + p.y.ToString("F2") + " z " + p.z.ToString("F2") + " yaw " +
            (p.yaw * 180.0 / System.Math.PI).ToString("F1") + " deg" + (cfg != null && cfg.noiseEnabled ? " | NOISE ON" : ""));
    }

    void OnDrawGizmos()
    {
        if (!drawGizmo || !haveOrigin || vehicle == null) return;
        PoseState p = Current;
        double c = System.Math.Cos(psi0), s = System.Math.Sin(psi0);
        // reported pose back into Unity world coordinates (for drawing only)
        float ux = (float)(x0 + p.x * c - p.y * s);
        float uz = (float)(z0 + p.x * s + p.y * c);
        Vector3 reportedPos = new Vector3(ux, vehicle.transform.position.y, uz);
        double psi = psi0 + p.yaw;
        Vector3 heading = new Vector3((float)System.Math.Cos(psi), 0f, (float)System.Math.Sin(psi));
        Gizmos.color = Color.blue;
        Gizmos.DrawSphere(vehicle.transform.position, 0.12f);
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(reportedPos, 0.12f);
        Gizmos.DrawLine(reportedPos, reportedPos + heading * 1.5f);
        Gizmos.color = Color.yellow;
        Gizmos.DrawLine(vehicle.transform.position, reportedPos);
    }
}