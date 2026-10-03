using UnityEngine;

// Kinematic bicycle model, rear-axle reference point.
// The GameObject origin is the rear axle centre at ground level.
// State: X, Z (Unity ground plane), Heading psi (rad, CCW from +X seen from above,
// left turns increase it), speed and steering angle (+ = left).
// Elevation and pitch follow the ground and do not affect the dynamics.
// Longitudinal: requested acceleration -> actuation delay -> clamp -> jerk limit -> speed.
[DefaultExecutionOrder(-100)] // runs before sensors and other scripts in the same physics step
public class Vehicle : MonoBehaviour
{
    [Header("Geometry")]
    public float wheelbase = 2.7f;      // m, rear axle to front axle
    public float maxSteeringDeg = 35f;
    public float maxSteeringRateDeg = 25f;  // deg/s, road-wheel angle rate limit

    [Header("Longitudinal dynamics")]
    public float maxAccel = 2.5f;           // m/s^2
    public float maxBrake = 8f;             // m/s^2 (positive number), one road surface
    public float actuationDelay = 0.10f;    // s, rounded to whole physics steps, set before Play
    public float jerkLimit = 80f;           // m/s^3, rate limit on acceleration

    [Header("Ground following")]
    public LayerMask groundMask;
    public float rayStartAbove = 3f;    // m above the current ground height
    public float rayLength = 13f;       // m

    [Header("Integration")]
    public bool midpointIntegration = true;

    double px, pz, psi;
    float groundY, pitch, speed, steer;
    float accel, jerk, requestedAccel;
    float[] delayLine;
    int delayIdx;
    bool warnedRear, warnedFront;

    public double X { get { return px; } }
    public double Z { get { return pz; } }
    public double Heading { get { return psi; } }

    // Initial condition / test setter. Normal operation changes speed through RequestedAccel.
    public float Speed
    {
        get { return speed; }
        set { speed = Mathf.Max(0f, value); }
    }

    float requestedSteer;

    // Immediate set for initial conditions and tests: sets the actual angle and the request.
    public float SteeringAngle
    {
        get { return steer; }
        set { steer = ClampSteer(value); requestedSteer = steer; }
    }

    // Requested steering angle (rad, + = left). The actual angle follows it at the rate limit.
    public float RequestedSteering
    {
        get { return requestedSteer; }
        set { requestedSteer = ClampSteer(value); }
    }

    float ClampSteer(float v)
    {
        float m = maxSteeringDeg * Mathf.Deg2Rad;
        return Mathf.Clamp(v, -m, m);
    }

    // Requested acceleration in m/s^2 (negative = braking). Stays in force until changed.
    public float RequestedAccel
    {
        get { return requestedAccel; }
        set { requestedAccel = value; }
    }

    public float Accel { get { return accel; } }   // actual acceleration after delay and jerk limit
    public float Jerk { get { return jerk; } }     // m/s^3 (not counted at the moment of stopping)
    public float YawRate { get { return (float)(speed / wheelbase * System.Math.Tan(steer)); } }
    public float PitchDeg { get { return pitch * Mathf.Rad2Deg; } }

    void Awake()
    {
        Vector3 p = transform.position;
        px = p.x;
        pz = p.z;
        groundY = p.y;
        psi = (90.0 - transform.eulerAngles.y) * System.Math.PI / 180.0;

        int n = Mathf.RoundToInt(actuationDelay / Time.fixedDeltaTime);
        delayLine = new float[Mathf.Max(0, n)];
    }

    public void SetPose(double x, double z, double headingRad)
    {
        px = x;
        pz = z;
        psi = headingRad;
        UpdateTransform();
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;

        // 1. actuation delay: the request from N steps ago reaches the actuators now
        float delayed = requestedAccel;
        if (delayLine.Length > 0)
        {
            delayed = delayLine[delayIdx];
            delayLine[delayIdx] = requestedAccel;
            delayIdx = (delayIdx + 1) % delayLine.Length;
        }
        float target = Mathf.Clamp(delayed, -maxBrake, maxAccel);

        // 2. jerk limit: acceleration moves toward the target at most jerkLimit * dt per step
        float aPrev = accel;
        float maxStep = jerkLimit * dt;
        accel = aPrev + Mathf.Clamp(target - aPrev, -maxStep, maxStep);
        jerk = (accel - aPrev) / dt;

        // 3. speed (trapezoid rule), no reversing
        float vOld = speed;
        float vNew = vOld + 0.5f * (aPrev + accel) * dt;
        if (vNew <= 0f)
        {
            vNew = 0f;
            if (accel < 0f) { accel = 0f; jerk = 0f; } // standstill is not a dynamic effect
        }
        speed = vNew;
        float vAvg = 0.5f * (vOld + vNew); // distance this step = vAvg * dt

        // 4. steering actuator: rate limit (road-wheel angle)
        float steerOld = steer;
        float maxSteerStep = maxSteeringRateDeg * Mathf.Deg2Rad * dt;
        steer = steerOld + Mathf.Clamp(requestedSteer - steerOld, -maxSteerStep, maxSteerStep);
        float steerMid = 0.5f * (steerOld + steer);

        // 5. planar motion (kinematic bicycle model)
        double w = vAvg / wheelbase * System.Math.Tan(steerMid); // yaw rate
        double pm = midpointIntegration ? psi + 0.5 * w * dt : psi;
        px += vAvg * System.Math.Cos(pm) * dt;
        pz += vAvg * System.Math.Sin(pm) * dt;
        psi += w * dt;

        UpdateTransform();
    }

    bool SampleGround(double x, double z, float refY, out float y)
    {
        var origin = new Vector3((float)x, refY + rayStartAbove, (float)z);
        RaycastHit hit;
        if (Physics.Raycast(origin, Vector3.down, out hit, rayLength, groundMask,
                            QueryTriggerInteraction.Ignore))
        {
            y = hit.point.y;
            return true;
        }
        y = refY;
        return false;
    }

    void UpdateTransform()
    {
        float yRear;
        if (SampleGround(px, pz, groundY, out yRear)) groundY = yRear;
        else if (!warnedRear)
        {
            warnedRear = true;
            UnityEngine.Debug.LogWarning("Vehicle: no ground below the rear axle (check Ground Mask and the layer of the ground objects)");
        }

        double fx = px + wheelbase * System.Math.Cos(psi);
        double fz = pz + wheelbase * System.Math.Sin(psi);
        float yFront;
        if (!SampleGround(fx, fz, groundY, out yFront))
        {
            yFront = groundY;
            if (!warnedFront)
            {
                warnedFront = true;
                UnityEngine.Debug.LogWarning("Vehicle: no ground below the front axle, pitch set to 0");
            }
        }
        pitch = Mathf.Atan2(yFront - groundY, wheelbase);

        double yawDeg = (90.0 - psi * (180.0 / System.Math.PI)) % 360.0;
        transform.SetPositionAndRotation(
            new Vector3((float)px, groundY, (float)pz),
            Quaternion.Euler(-pitch * Mathf.Rad2Deg, (float)yawDeg, 0f));
    }

    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(transform.position, 0.15f);
        Gizmos.color = Color.green;
        Gizmos.DrawSphere(transform.position + transform.forward * wheelbase, 0.15f);
    }
}