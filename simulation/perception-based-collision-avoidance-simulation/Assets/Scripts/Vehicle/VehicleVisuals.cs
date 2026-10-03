using UnityEngine;

// Cosmetic vehicle visuals: wheel spin, front wheel steering (Ackermann), body pitch.
// Purely visual. Sensors are mounted on the Vehicle root (kinematic frame), not on these objects.
[DefaultExecutionOrder(10)] // after Vehicle (-100)
public class VehicleVisuals : MonoBehaviour
{
    public Vehicle vehicle;
    public Transform rig;                    // only used in the report below

    [Header("Wheels (left = the negative-X side of the car)")]
    public Transform frontLeft;
    public Transform frontRight;
    public Transform rearLeft;
    public Transform rearRight;
    public float wheelRadius = 0.34f;        // m, taken from "Measure and report"
    public bool ackermann = true;

    [Header("Cosmetic body pitch (nose dips when braking)")]
    public Transform bodyPivot;
    public float pitchPerAccelDeg = 0.4f;    // deg per m/s^2
    public float maxBodyPitchDeg = 3f;
    public float pitchTimeConstant = 0.15f;  // s

    [Header("Read-only (debug)")]
    public float leftSteerDeg;
    public float rightSteerDeg;
    public float bodyPitchDeg;

    Transform[] w;
    Quaternion[] rest;
    float spinDeg;
    float frontTrack;

    static Vector3 Center(Transform t)
    {
        Renderer r = t.GetComponentInChildren<Renderer>();
        return r != null ? r.bounds.center : t.position;
    }

    void Start()
    {
        w = new[] { frontLeft, frontRight, rearLeft, rearRight };
        rest = new Quaternion[4];
        for (int i = 0; i < 4; i++) rest[i] = w[i].localRotation;
        frontTrack = Vector3.Distance(Center(frontLeft), Center(frontRight));
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;

        // wheel spin: rolling without slip, omega = v / r
        spinDeg = Mathf.Repeat(spinDeg + vehicle.Speed / wheelRadius * dt * Mathf.Rad2Deg, 360f);

        // steering angles of the front wheels (+ = left)
        float d = vehicle.SteeringAngle;
        float t = Mathf.Tan(d);
        float L = vehicle.wheelbase;
        float dl = d, dr = d;
        if (ackermann)
        {
            float half = 0.5f * frontTrack;
            dl = Mathf.Atan(L * t / (L - half * t));
            dr = Mathf.Atan(L * t / (L + half * t));
        }
        leftSteerDeg = dl * Mathf.Rad2Deg;
        rightSteerDeg = dr * Mathf.Rad2Deg;

        // Unity yaw is positive to the right, so a left steering angle is a negative yaw
        Apply(0, -leftSteerDeg);
        Apply(1, -rightSteerDeg);
        Apply(2, 0f);
        Apply(3, 0f);

        // cosmetic body pitch: braking (negative acceleration) dips the nose (+X rotation is nose down)
        float target = Mathf.Clamp(-vehicle.Accel * pitchPerAccelDeg, -maxBodyPitchDeg, maxBodyPitchDeg);
        bodyPitchDeg += (target - bodyPitchDeg) * (1f - Mathf.Exp(-dt / pitchTimeConstant));
        if (bodyPivot != null) bodyPivot.localRotation = Quaternion.Euler(bodyPitchDeg, 0f, 0f);
    }

    // steer about the vertical axis, spin about the car's lateral axis, on top of the imported rotation
    void Apply(int i, float steerYawDeg)
    {
        w[i].localRotation = Quaternion.AngleAxis(steerYawDeg, Vector3.up)
                           * Quaternion.AngleAxis(spinDeg, Vector3.right)
                           * rest[i];
    }

    // Edit-mode helper: measures the model in Vehicle space (meters). Keep the Vehicle at rotation 0.
    [ContextMenu("Measure and report")]
    void Measure()
    {
        Transform v = vehicle.transform;
        Vector3 fl = v.InverseTransformPoint(Center(frontLeft));
        Vector3 fr = v.InverseTransformPoint(Center(frontRight));
        Vector3 rl = v.InverseTransformPoint(Center(rearLeft));
        Vector3 rr = v.InverseTransformPoint(Center(rearRight));
        Vector3 fm = 0.5f * (fl + fr);
        Vector3 rm = 0.5f * (rl + rr);

        float radius = 0f;
        foreach (Transform wt in new[] { frontLeft, frontRight, rearLeft, rearRight })
        {
            Renderer r = wt.GetComponentInChildren<Renderer>();
            if (r != null) radius += 0.25f * r.bounds.extents.y;
        }

        Bounds b = new Bounds();
        bool first = true;
        foreach (Renderer r in GetComponentsInChildren<Renderer>())
        {
            if (first) { b = r.bounds; first = false; }
            else b.Encapsulate(r.bounds);
        }

        Vector3 shift = new Vector3(-rm.x, radius - rm.y, -rm.z);
        UnityEngine.Debug.Log(
            "VEHICLE MODEL REPORT (meters, Vehicle space)\n" +
            "wheelbase " + (fm.z - rm.z).ToString("F3") + " | front track " + Mathf.Abs(fl.x - fr.x).ToString("F3") +
            " | rear track " + Mathf.Abs(rl.x - rr.x).ToString("F3") + " | wheel radius " + radius.ToString("F3") + "\n" +
            "rear axle centre (" + rm.x.ToString("F3") + ", " + rm.y.ToString("F3") + ", " + rm.z.ToString("F3") + ")" +
            " | front axle centre (" + fm.x.ToString("F3") + ", " + fm.y.ToString("F3") + ", " + fm.z.ToString("F3") + ")\n" +
            "model length " + b.size.z.ToString("F2") + " | width " + b.size.x.ToString("F2") + " | height " + b.size.y.ToString("F2") + "\n" +
            "To put the rear axle at (0, radius, 0), add to the Rig position: (" + shift.x.ToString("F3") + ", " +
            shift.y.ToString("F3") + ", " + shift.z.ToString("F3") + ")  |  scale factor for a 2.7 m wheelbase: " +
            (vehicle.wheelbase / Mathf.Max(0.001f, fm.z - rm.z)).ToString("F3") + " (relative to the current Rig scale)");
    }
}