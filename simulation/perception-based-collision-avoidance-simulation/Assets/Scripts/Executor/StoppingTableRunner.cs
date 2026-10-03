using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public class StoppingTableRunner : MonoBehaviour
{
    public Vehicle vehicle;
    public CommandExecutor executor;
    public float[] speedsKmh = { 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    public float settleTime = 1f;                       // s at steady speed before the command
    public string outputFolder = "experiments/phase2";  // relative to the repository root

    enum Phase { Setup, Settle, Braking, Finished }
    Phase phase = Phase.Setup;
    int index;
    long steps, settleSteps;
    double x0, z0, tCmd;
    float v0, peakDecel, peakJerk;
    readonly StringBuilder csv = new StringBuilder("run,kmh,v0_ms,distance_m,time_s,peak_decel,peak_jerk\n");

    void FixedUpdate()
    {
        steps++;
        double t = steps * (double)Time.fixedDeltaTime;

        switch (phase)
        {
            case Phase.Setup:
                if (index >= speedsKmh.Length) { Finish(); return; }
                executor.Clear();
                vehicle.ResetLongitudinal();
                vehicle.SteeringAngle = 0f;
                vehicle.SetPose(0.0, 0.0, System.Math.PI / 2.0); // start of the road, heading +Z
                vehicle.Speed = speedsKmh[index] / 3.6f;
                settleSteps = 0;
                phase = Phase.Settle;
                break;

            case Phase.Settle:
                settleSteps++;
                if (settleSteps * Time.fixedDeltaTime >= settleTime)
                {
                    v0 = vehicle.Speed;
                    x0 = vehicle.X;
                    z0 = vehicle.Z;
                    tCmd = t;
                    peakDecel = 0f;
                    peakJerk = 0f;
                    executor.Submit(new VehicleCommand
                    {
                        id = (uint)(index + 1),
                        dataTimestamp = t,
                        comfortCap = 99f,
                        safetyCap = 99f,
                        comfortDecel = 3f,
                        emergency = true,
                        validity = 60f,
                        riskLevel = 1
                    });
                    phase = Phase.Braking;
                }
                break;

            case Phase.Braking:
                peakDecel = Mathf.Max(peakDecel, -vehicle.Accel);
                peakJerk = Mathf.Max(peakJerk, Mathf.Abs(vehicle.Jerk));
                if (vehicle.Speed <= 0f)
                {
                    double dx = vehicle.X - x0, dz = vehicle.Z - z0;
                    double dist = System.Math.Sqrt(dx * dx + dz * dz);
                    double time = t - tCmd;
                    csv.Append(string.Format(CultureInfo.InvariantCulture, "{0},{1:F0},{2:F4},{3:F4},{4:F3},{5:F3},{6:F2}\n",
                        index + 1, speedsKmh[index], v0, dist, time, peakDecel, peakJerk));
                    UnityEngine.Debug.Log(speedsKmh[index].ToString("F0") + " km/h (" + v0.ToString("F3") + " m/s): stopped after " +
                        dist.ToString("F3") + " m in " + time.ToString("F2") + " s | peak decel " + peakDecel.ToString("F2") +
                        " | peak jerk " + peakJerk.ToString("F1"));
                    index++;
                    phase = Phase.Setup;
                }
                break;
        }
    }

    void Finish()
    {
        phase = Phase.Finished;
        string root = Directory.GetParent(ConfigPaths.ConfigDir()).FullName;
        string dir = Path.Combine(root, outputFolder);
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "stopping_table.csv");
        File.WriteAllText(path, csv.ToString());
        UnityEngine.Debug.Log("Stopping table written to " + path);
    }
}