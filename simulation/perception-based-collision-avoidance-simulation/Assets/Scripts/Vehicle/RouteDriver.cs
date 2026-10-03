using UnityEngine;

// Scripted driver: pure-pursuit steering and speed-profile following on a route.
// It never reacts to obstacles. In 2.6 the command executor will sit between this and the vehicle.
public class RouteDriver : MonoBehaviour
{
    public Vehicle vehicle;
    public string routeFile = "routes/curves.json";

    [Header("Speed")]
    public float cruiseSpeed = 14f;        // m/s, scripted speed
    public float maxLateralAccel = 3f;     // m/s^2, sets curve speeds
    public float comfortDecel = 2f;        // m/s^2, slowing before curves and the end
    public float comfortAccel = 1.5f;      // m/s^2
    public float speedGain = 1f;           // 1/s, P gain on the speed error
    public float driverMaxBrake = 3f;      // m/s^2, the driver never uses emergency braking

    [Header("Pure pursuit")]
    public float lookaheadMin = 2f;        // m
    public float lookaheadPerSpeed = 0.4f; // s
    public float lookaheadMax = 15f;       // m
    public float endMargin = 0.5f;         // m before the route end where the run finishes

    [Header("Logging")]
    public float logInterval = 5f;         // s of simulation time

    public Route Track { get; private set; }
    public float S { get; private set; }            // arc length of the closest route point
    public float CrossTrack { get; private set; }   // m, + = vehicle is left of the route
    public float TargetSpeed { get; private set; }  // m/s, profile speed at S
    public bool Finished { get; private set; }

    SpeedProfile profile;
    long steps;
    double nextLog;
    bool reported;
    float peakSpeed, maxCross, maxLat, maxSteer, maxSteerRate, prevSteer;

    void Start()
    {
        Track = RouteIO.Build(RouteIO.Read(ConfigPaths.Resolve(routeFile)));
        profile = new SpeedProfile(Track, cruiseSpeed, maxLateralAccel, comfortDecel, comfortAccel);
        RouteSample s0 = Track.Sample(0f);
        vehicle.SetPose(s0.position.x, s0.position.z, s0.heading);
        vehicle.Speed = 0f;
        nextLog = logInterval;
    }

    void FixedUpdate()
    {
        if (Track == null) return;
        steps++;
        double t = steps * (double)Time.fixedDeltaTime;

        double px = vehicle.X, pz = vehicle.Z;
        S = Track.FindClosestS(px, pz, S);
        RouteSample c = Track.Sample(S);

        // signed cross-track error: + = left of the route
        CrossTrack = (float)(-Mathf.Sin(c.heading) * (px - c.position.x)
                             + Mathf.Cos(c.heading) * (pz - c.position.z));

        // statistics
        peakSpeed = Mathf.Max(peakSpeed, vehicle.Speed);
        maxCross = Mathf.Max(maxCross, Mathf.Abs(CrossTrack));
        maxLat = Mathf.Max(maxLat, Mathf.Abs(vehicle.Speed * vehicle.YawRate));
        maxSteer = Mathf.Max(maxSteer, Mathf.Abs(vehicle.SteeringAngle));
        maxSteerRate = Mathf.Max(maxSteerRate, Mathf.Abs(vehicle.SteeringAngle - prevSteer) / Time.fixedDeltaTime);
        prevSteer = vehicle.SteeringAngle;

        if (t >= nextLog)
        {
            nextLog += logInterval;
            UnityEngine.Debug.Log("t=" + t.ToString("F1") + " s | s=" + S.ToString("F1") + " m | v=" +
                vehicle.Speed.ToString("F2") + " (profile " + TargetSpeed.ToString("F2") + ") m/s | cross-track " +
                CrossTrack.ToString("F3") + " m | steering " + (vehicle.SteeringAngle * Mathf.Rad2Deg).ToString("F2") + " deg");
        }

        // end of route: straighten and stop
        if (Finished || S >= Track.Length - endMargin)
        {
            Finished = true;
            vehicle.RequestedSteering = 0f;
            vehicle.RequestedAccel = -driverMaxBrake;
            if (!reported && vehicle.Speed <= 0f)
            {
                reported = true;
                UnityEngine.Debug.Log("FINISHED after " + t.ToString("F2") + " s | stopped at s=" + S.ToString("F2") +
                    " of " + Track.Length.ToString("F2") + " m | peak speed " + peakSpeed.ToString("F2") +
                    " m/s | max |cross-track| " + maxCross.ToString("F3") + " m | max lateral accel " +
                    maxLat.ToString("F2") + " m/s^2 | max steering " + (maxSteer * Mathf.Rad2Deg).ToString("F1") +
                    " deg | max steering rate " + (maxSteerRate * Mathf.Rad2Deg).ToString("F1") + " deg/s");
            }
            return;
        }

        // pure pursuit steering
        float ld = Mathf.Clamp(lookaheadMin + lookaheadPerSpeed * vehicle.Speed, lookaheadMin, lookaheadMax);
        RouteSample tgt = Track.Sample(Mathf.Min(S + ld, Track.Length));
        double dx = tgt.position.x - px, dz = tgt.position.z - pz;
        double chord = System.Math.Max(System.Math.Sqrt(dx * dx + dz * dz), 0.5);
        double alpha = System.Math.IEEERemainder(System.Math.Atan2(dz, dx) - vehicle.Heading, 2.0 * System.Math.PI);
        vehicle.RequestedSteering = (float)System.Math.Atan(2.0 * vehicle.wheelbase * System.Math.Sin(alpha) / chord);

        // speed: profile feed-forward plus P correction
        float vTarget, aFf;
        profile.At(S, out vTarget, out aFf);
        TargetSpeed = vTarget;
        float aReq = aFf + speedGain * (vTarget - vehicle.Speed);
        vehicle.RequestedAccel = Mathf.Clamp(aReq, -driverMaxBrake, vehicle.maxAccel);
    }
}