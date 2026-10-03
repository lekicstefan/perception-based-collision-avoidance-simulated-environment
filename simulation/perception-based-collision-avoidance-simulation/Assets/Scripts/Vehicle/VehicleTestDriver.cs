using UnityEngine;

public class VehicleTestDriver : MonoBehaviour
{
    public Vehicle vehicle;
    public float speed = 5f;            // m/s
    public float steeringDeg = 5.7f;    // + = left
    public bool placeAtRouteStart = false;
    public string routeFile = "routes/hill_straight.json";
    public float logInterval = 10f;     // s of simulation time

    double x0, z0, psi0, nextLog;
    long steps;

    void Start()
    {
        if (placeAtRouteStart)
        {
            Route r = RouteIO.Build(RouteIO.Read(ConfigPaths.Resolve(routeFile)));
            RouteSample s0 = r.Sample(0f);
            vehicle.SetPose(s0.position.x, s0.position.z, s0.heading);
        }
        vehicle.Speed = speed;
        vehicle.SteeringAngle = steeringDeg * Mathf.Deg2Rad;
        x0 = vehicle.X;
        z0 = vehicle.Z;
        psi0 = vehicle.Heading;
        nextLog = logInterval;
    }

    void FixedUpdate()
    {
        steps++;
        double t = steps * (double)Time.fixedDeltaTime;
        if (t >= nextLog)
        {
            nextLog += logInterval;
            Log(t);
        }
    }

    void Log(double t)
    {
        double v = vehicle.Speed, d = vehicle.SteeringAngle, L = vehicle.wheelbase;
        double ex, ez, epsi;
        if (System.Math.Abs(d) < 1e-6)
        {
            epsi = psi0;
            ex = x0 + v * t * System.Math.Cos(psi0);
            ez = z0 + v * t * System.Math.Sin(psi0);
        }
        else
        {
            double w = v / L * System.Math.Tan(d);
            double R = v / w; // signed radius, + = left
            epsi = psi0 + w * t;
            ex = x0 + R * (System.Math.Sin(epsi) - System.Math.Sin(psi0));
            ez = z0 - R * (System.Math.Cos(epsi) - System.Math.Cos(psi0));
        }
        double posErrMm = 1000.0 * System.Math.Sqrt((vehicle.X - ex) * (vehicle.X - ex) + (vehicle.Z - ez) * (vehicle.Z - ez));
        double headErrDeg = (vehicle.Heading - epsi) * 180.0 / System.Math.PI;
        double travelled = System.Math.Sqrt((vehicle.X - x0) * (vehicle.X - x0) + (vehicle.Z - z0) * (vehicle.Z - z0));
        UnityEngine.Debug.Log(
            "t=" + t.ToString("F2") + " s | position error " + posErrMm.ToString("F4") + " mm | heading error " +
            headErrDeg.ToString("F5") + " deg | straight-line distance from start " + travelled.ToString("F1") +
            " m | elevation " + vehicle.transform.position.y.ToString("F3") + " m | pitch " +
            vehicle.PitchDeg.ToString("F2") + " deg");
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 700, 25),
            "v=" + vehicle.Speed.ToString("F1") + " m/s   pitch=" + vehicle.PitchDeg.ToString("F2") +
            " deg   elevation=" + vehicle.transform.position.y.ToString("F2") + " m");
    }
}