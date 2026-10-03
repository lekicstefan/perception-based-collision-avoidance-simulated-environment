using UnityEngine;

public class LongitudinalTestDriver : MonoBehaviour
{
    public enum Mode { Brake, Accelerate }

    public Vehicle vehicle;
    public Mode mode = Mode.Brake;
    public float initialSpeed = 20f;      // m/s, Brake mode
    public float requestedAccel = -8f;    // m/s^2; beyond the vehicle's limits it is clamped
    public float requestTime = 2f;        // s of simulation time before the request
    public float accelerateDuration = 6f; // s, Accelerate mode

    long steps;
    bool requested, finished;
    double tReq, x0, z0;
    float v0, tFirst = -1f, tReach = -1f, peakAccel, peakJerk;

    void Start()
    {
        if (mode == Mode.Brake) vehicle.Speed = initialSpeed;
    }

    void FixedUpdate()
    {
        if (finished) return;
        steps++;
        double t = steps * (double)Time.fixedDeltaTime;

        if (!requested)
        {
            if (t >= requestTime)
            {
                requested = true;
                tReq = t;
                x0 = vehicle.X;
                z0 = vehicle.Z;
                v0 = vehicle.Speed;
                vehicle.RequestedAccel = requestedAccel;
            }
            return;
        }

        float mag = Mathf.Abs(vehicle.Accel);
        float target = Mathf.Abs(Mathf.Clamp(requestedAccel, -vehicle.maxBrake, vehicle.maxAccel));
        float rel = (float)(t - tReq);
        if (tFirst < 0f && mag > 0.01f) tFirst = rel;
        if (tReach < 0f && target > 0f && mag >= 0.99f * target) tReach = rel;
        peakAccel = Mathf.Max(peakAccel, mag);
        peakJerk = Mathf.Max(peakJerk, Mathf.Abs(vehicle.Jerk));

        bool done = (mode == Mode.Brake) ? vehicle.Speed <= 0f : rel >= accelerateDuration;
        if (done)
        {
            finished = true;
            Report(rel);
        }
    }

    void Report(float rel)
    {
        double dist = System.Math.Sqrt((vehicle.X - x0) * (vehicle.X - x0) + (vehicle.Z - z0) * (vehicle.Z - z0));
        string common = "first response after " + tFirst.ToString("F2") + " s | 99% of target after " +
                        tReach.ToString("F2") + " s | peak accel " + peakAccel.ToString("F2") +
                        " m/s^2 | peak jerk " + peakJerk.ToString("F1") + " m/s^3";
        if (mode == Mode.Brake)
        {
            double a = System.Math.Min(System.Math.Abs(requestedAccel), vehicle.maxBrake);
            double tr = a / vehicle.jerkLimit;
            double teffModel = vehicle.actuationDelay + tr / 2.0;
            double teffFit = (dist - v0 * v0 / (2.0 * a) + a * tr * tr / 24.0) / v0;
            UnityEngine.Debug.Log("BRAKE v0=" + v0.ToString("F1") + " m/s, request=" + requestedAccel.ToString("F1") +
                " | " + common + " | stopped after " + rel.ToString("F2") + " s, distance " + dist.ToString("F3") +
                " m | fitted t_eff " + teffFit.ToString("F4") + " s (t_d + t_r/2 = " + teffModel.ToString("F4") + " s)");
        }
        else
        {
            UnityEngine.Debug.Log("ACCELERATE request=" + requestedAccel.ToString("F1") + " | " + common +
                " | speed after " + rel.ToString("F1") + " s: " + vehicle.Speed.ToString("F3") + " m/s");
        }
    }
}