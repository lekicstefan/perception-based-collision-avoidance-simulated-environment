using System.Text;
using UnityEngine;

public class PoseTest : MonoBehaviour
{
    public string configName = "pose_noise";

    static string Check(bool ok) { return ok ? "PASS" : "FAIL"; }
    static bool Near(double a, double b) { return System.Math.Abs(a - b) < 1e-9; }

    [ContextMenu("Run pose tests")]
    void Run()
    {
        const double D = System.Math.PI / 180.0;
        var sb = new StringBuilder("POSE TESTS\n");
        double xr, yr, yaw;

        // start heading +Z (psi0 = 90 deg): left is -X
        PoseEstimator.ToStartFrame(5, 7, 90 * D, 5, 17, 90 * D, out xr, out yr, out yaw);
        sb.AppendLine("10 m forward                 : " + Check(Near(xr, 10) && Near(yr, 0) && Near(yaw, 0)));
        PoseEstimator.ToStartFrame(5, 7, 90 * D, 2, 7, 90 * D, out xr, out yr, out yaw);
        sb.AppendLine("3 m to the left (-X)         : " + Check(Near(xr, 0) && Near(yr, 3)));
        PoseEstimator.ToStartFrame(5, 7, 90 * D, 5, 7, 180 * D, out xr, out yr, out yaw);
        sb.AppendLine("turned 90 deg to the left    : " + Check(Near(yaw, 90 * D)));
        // start heading +X (psi0 = 0): left is +Z
        PoseEstimator.ToStartFrame(1, 1, 0, 6, 1, 0, out xr, out yr, out yaw);
        sb.AppendLine("start facing +X, 5 m ahead   : " + Check(Near(xr, 5) && Near(yr, 0)));
        PoseEstimator.ToStartFrame(1, 1, 0, 1, 3, 0, out xr, out yr, out yaw);
        sb.AppendLine("start facing +X, 2 m left    : " + Check(Near(xr, 0) && Near(yr, 2)));
        PoseEstimator.ToStartFrame(0, 0, 90 * D, 0, 0, 90 * D + 3 * System.Math.PI, out xr, out yr, out yaw);
        sb.AppendLine("yaw wrapped to (-pi, pi]     : " + Check(Near(System.Math.Abs(yaw), System.Math.PI)));

        PoseConfig cfg = SensorConfigLoader.Load(configName).pose;
        const double dt = 0.01;

        // white noise from successive differences of one long run
        var m = new PoseNoiseModel(cfg, 12345);
        int steps = 20000;
        double sx = 0, syaw = 0, px = 0, pyaw = 0;
        for (int k = 0; k < steps; k++)
        {
            double dx, dy, dyaw;
            m.Step(dt, out dx, out dy, out dyaw);
            if (k > 0) { sx += (dx - px) * (dx - px); syaw += (dyaw - pyaw) * (dyaw - pyaw); }
            px = dx; pyaw = dyaw;
        }
        sb.AppendLine("white noise: position std " + System.Math.Sqrt(sx / (steps - 1) / 2).ToString("F4") + " m (config " + cfg.positionNoiseStdM +
            "), yaw std " + (System.Math.Sqrt(syaw / (steps - 1) / 2) / D).ToString("F4") + " deg (config " + cfg.yawNoiseStdDeg + ")");

        // bias after 100 s over 400 independent runs
        int runs = 400, n = 10000;
        double bx2 = 0, byaw2 = 0;
        for (int r = 0; r < runs; r++)
        {
            var mm = new PoseNoiseModel(cfg, 1000 + r);
            double a, b, c;
            for (int k = 0; k < n; k++) mm.Step(dt, out a, out b, out c);
            bx2 += mm.BiasX * mm.BiasX;
            byaw2 += mm.BiasYaw * mm.BiasYaw;
        }
        sb.AppendLine("bias after 100 s: position std " + System.Math.Sqrt(bx2 / runs).ToString("F3") + " m (expected " +
            (cfg.positionBiasWalkMPerSqrtS * 10.0).ToString("F3") + "), yaw std " + (System.Math.Sqrt(byaw2 / runs) / D).ToString("F3") +
            " deg (expected " + (cfg.yawBiasWalkDegPerSqrtS * 10.0).ToString("F3") + ")");

        // reproducibility
        var a1 = new PoseNoiseModel(cfg, 777);
        var a2 = new PoseNoiseModel(cfg, 777);
        var a3 = new PoseNoiseModel(cfg, 778);
        bool same = true, differs = false;
        for (int k = 0; k < 1000; k++)
        {
            double x1, y1, w1, x2, y2, w2, x3, y3, w3;
            a1.Step(dt, out x1, out y1, out w1);
            a2.Step(dt, out x2, out y2, out w2);
            a3.Step(dt, out x3, out y3, out w3);
            if (x1 != x2 || y1 != y2 || w1 != w2) same = false;
            if (x1 != x3) differs = true;
        }
        sb.AppendLine("same seed gives the same stream: " + Check(same) + " | different seed differs: " + Check(differs));
        UnityEngine.Debug.Log(sb.ToString());
    }
}