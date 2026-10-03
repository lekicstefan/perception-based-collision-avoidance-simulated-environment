using UnityEngine;

public class SteeringStepTest : MonoBehaviour
{
    public Vehicle vehicle;
    public float stepDeg = 20f;
    public float stepTime = 1f;

    long steps;
    bool applied, done;
    double tApplied;
    float prev, maxRate;

    void FixedUpdate()
    {
        if (done) return;
        steps++;
        double t = steps * (double)Time.fixedDeltaTime;

        if (!applied)
        {
            if (t >= stepTime)
            {
                applied = true;
                tApplied = t;
                prev = vehicle.SteeringAngle;
                vehicle.RequestedSteering = stepDeg * Mathf.Deg2Rad;
            }
            return;
        }

        float cur = vehicle.SteeringAngle;
        maxRate = Mathf.Max(maxRate, Mathf.Abs(cur - prev) / Time.fixedDeltaTime);
        prev = cur;
        float target = vehicle.RequestedSteering;
        if (Mathf.Abs(cur) >= 0.99f * Mathf.Abs(target))
        {
            done = true;
            UnityEngine.Debug.Log("STEERING STEP requested " + stepDeg.ToString("F1") + " deg, target after clamp " +
                (target * Mathf.Rad2Deg).ToString("F1") + " deg | 99% reached after " + (t - tApplied).ToString("F2") +
                " s | peak rate " + (maxRate * Mathf.Rad2Deg).ToString("F1") + " deg/s");
        }
    }
}