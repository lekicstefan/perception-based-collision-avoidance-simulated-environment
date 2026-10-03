using UnityEngine;

public class ExecutorTestScript : MonoBehaviour
{
    public enum Test { ComfortCap, SafetyCap, BrakeRequest, Emergency, ExpiryFallback }

    public CommandExecutor executor;
    public Vehicle vehicle;
    public Test test = Test.ComfortCap;
    public float startTime = 20f;       // s: the car cruises at the driver's speed until then
    public float capValue = 8f;         // m/s (cap tests)
    public float comfortDecel = 3f;     // m/s^2
    public float brakeRequest = 5f;     // m/s^2 (BrakeRequest test)
    public float validity = 2f;         // s (ExpiryFallback test)
    public float fallbackDecel = 3f;    // m/s^2 (ExpiryFallback test)

    long steps;
    bool submitted, finished;
    double tSub, xSub, zSub;
    float vSub, peakDecel;

    double Dist(double x0, double z0)
    {
        double dx = vehicle.X - x0, dz = vehicle.Z - z0;
        return System.Math.Sqrt(dx * dx + dz * dz);
    }

    void SubmitCommand()
    {
        var c = new VehicleCommand
        {
            id = 1,
            dataTimestamp = Time.fixedTimeAsDouble,
            comfortCap = 99f,
            safetyCap = 99f,
            comfortDecel = comfortDecel,
            validity = 60f,
            riskLevel = 1
        };
        switch (test)
        {
            case Test.ComfortCap: c.comfortCap = capValue; break;
            case Test.SafetyCap: c.safetyCap = capValue; break;
            case Test.BrakeRequest: c.brakeRequest = brakeRequest; break;
            case Test.Emergency: c.emergency = true; break;
            case Test.ExpiryFallback: c.validity = validity; c.fallbackDecel = fallbackDecel; break;
        }
        executor.Submit(c);
    }

    void Done(string text)
    {
        finished = true;
        UnityEngine.Debug.Log("TEST " + test + " | speed at submit " + vSub.ToString("F2") + " m/s | " + text +
                              " | peak decel " + peakDecel.ToString("F2") + " m/s^2");
    }

    void FixedUpdate()
    {
        if (finished) return;
        steps++;
        double t = steps * (double)Time.fixedDeltaTime;

        if (!submitted)
        {
            if (t >= startTime)
            {
                SubmitCommand();
                submitted = true;
                tSub = t; xSub = vehicle.X; zSub = vehicle.Z; vSub = vehicle.Speed;
            }
            return;
        }

        peakDecel = Mathf.Max(peakDecel, -vehicle.Accel);
        double rel = t - tSub;
        switch (test)
        {
            case Test.ComfortCap:
            case Test.SafetyCap:
                if (Mathf.Abs(vehicle.Speed - capValue) < 0.1f)
                    Done("cap reached (within 0.1 m/s) after " + rel.ToString("F2") + " s, " + Dist(xSub, zSub).ToString("F2") + " m");
                break;
            case Test.BrakeRequest:
            case Test.Emergency:
                if (vehicle.Speed <= 0f)
                    Done("stopped after " + rel.ToString("F2") + " s, " + Dist(xSub, zSub).ToString("F2") + " m");
                break;
            case Test.ExpiryFallback:
                if (executor.Status == CommandExecutor.CommandStatus.Expired && vehicle.Speed <= 0f)
                    Done("expired " + (executor.ExpiredAt - executor.AppliedAt).ToString("F2") + " s after it was applied; stopped " +
                         (t - executor.ExpiredAt).ToString("F2") + " s after expiry, " +
                         Dist(executor.ExpiredX, executor.ExpiredZ).ToString("F2") + " m after expiry");
                break;
        }
    }
}