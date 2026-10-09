using System.Text;
using UnityEngine;

public class CalibrationTest : MonoBehaviour
{
    public string configName = "default";
    public Vehicle vehicle;
    public BoxCollider egoBox;      // the CollisionBox

    static string Check(bool ok) { return ok ? "PASS" : "FAIL"; }
    static string Fmt(double[] a)
    {
        var s = new StringBuilder();
        foreach (double d in a) s.Append(d.ToString("F3")).Append(' ');
        return s.ToString();
    }

    [ContextMenu("Run calibration tests")]
    void Run()
    {
        var sb = new StringBuilder("CALIBRATION TESTS\n");
        SensorConfig cfg = SensorConfigLoader.Load(configName);
        CalibrationData c = CalibrationBuilder.Build(cfg, vehicle, egoBox);
        string json = CalibrationBuilder.ToJson(c, false);
        EgoCalibration e = c.ego;
        CameraCalibration cam = c.camera;
        LidarCalibration lid = c.lidar;

        sb.AppendLine("id " + c.calibrationId + ", " + json.Length + " bytes of JSON");
        sb.AppendLine("ego: length " + e.length.ToString("F2") + ", width " + e.width.ToString("F2") + ", height " + e.height.ToString("F2") +
            ", wheelbase " + e.wheelbase.ToString("F2") + ", rear overhang " + e.rearOverhang.ToString("F2") + ", front overhang " + e.frontOverhang.ToString("F2"));
        sb.AppendLine("camera: fx " + cam.fx.ToString("F1") + " fy " + cam.fy.ToString("F1") + " cx " + cam.cx.ToString("F1") + " cy " + cam.cy.ToString("F1") +
            " vFOV " + cam.verticalFovDeg.ToString("F2"));
        sb.AppendLine("  optical R: " + Fmt(cam.opticalFromVehicle.R) + "| t: " + Fmt(cam.opticalFromVehicle.t));
        sb.AppendLine("lidar: " + lid.rows + " x " + lid.cols + ", elevation " + lid.elevationsDeg[0].ToString("F2") + " .. " +
            lid.elevationsDeg[lid.rows - 1].ToString("F2") + ", azimuth " + lid.azimuthsDeg[0].ToString("F2") + " .. " + lid.azimuthsDeg[lid.cols - 1].ToString("F2"));

        // the two test cubes of step 3.3: Unity (-3, 1, 15) and (3, 1, 15) -> vehicle frame (15, 3, 1) and (15, -3, 1)
        double u, v, d;
        CalibrationBuilder.Project(cam, 15, 3, 1, out u, out v, out d);
        sb.AppendLine("red cube   -> pixel (" + u.ToString("F2") + ", " + v.ToString("F2") + "), depth " + d.ToString("F2") + "  (expected 247.27, 187.27, 13.20)");
        CalibrationBuilder.Project(cam, 15, -3, 1, out u, out v, out d);
        sb.AppendLine("green cube -> pixel (" + u.ToString("F2") + ", " + v.ToString("F2") + "), depth " + d.ToString("F2") + "  (expected 392.73, 187.27, 13.20)");

        // mount rotation: matrix against Unity's quaternion path
        var m = new Mount { x = 1.8f, y = 0.2f, z = 1.3f, yawDeg = 30f, pitchDeg = 10f, rollDeg = 20f };
        double[,] r = CalibrationBuilder.RotationFromMount(m);
        Quaternion q = Frames.MountRotationToUnity(m.yawDeg, m.pitchDeg, m.rollDeg);
        Vector3[] bodyAxes = { Vector3.forward, Vector3.left, Vector3.up };   // body x (forward), y (left), z (up) in Unity
        string[] names = { "forward", "left", "up" };
        double maxDiff = 0.0;
        for (int k = 0; k < 3; k++)
        {
            float px, py, pz;
            Frames.FromUnity(q * bodyAxes[k], out px, out py, out pz);
            maxDiff = System.Math.Max(maxDiff, System.Math.Max(System.Math.Abs(px - r[0, k]), System.Math.Max(System.Math.Abs(py - r[1, k]), System.Math.Abs(pz - r[2, k]))));
            sb.AppendLine("  " + names[k] + " axis (yaw 30, pitch 10, roll 20): Unity path " + px.ToString("F4") + " " + py.ToString("F4") + " " + pz.ToString("F4") +
                " | matrix " + r[0, k].ToString("F4") + " " + r[1, k].ToString("F4") + " " + r[2, k].ToString("F4"));
        }
        sb.AppendLine("mount rotation, matrix vs Unity quaternion: " + Check(maxDiff < 1e-5));

        // projection through that rotated mount
        cfg.camera.mount = m;
        CalibrationData c2 = CalibrationBuilder.Build(cfg, vehicle, egoBox);
        double tx = m.x, ty = m.y, tz = m.z;
        CalibrationBuilder.Project(c2.camera, tx + 10 * r[0, 0], ty + 10 * r[1, 0], tz + 10 * r[2, 0], out u, out v, out d);
        sb.AppendLine("10 m along the optical axis -> (" + u.ToString("F3") + ", " + v.ToString("F3") + ", " + d.ToString("F3") + ")  " +
            Check(System.Math.Abs(u - c2.camera.cx) < 1e-3 && System.Math.Abs(v - c2.camera.cy) < 1e-3 && System.Math.Abs(d - 10) < 1e-6));
        CalibrationBuilder.Project(c2.camera, tx + 10 * r[0, 0] + r[0, 1], ty + 10 * r[1, 0] + r[1, 1], tz + 10 * r[2, 0] + r[2, 1], out u, out v, out d);
        sb.AppendLine("+ 1 m to the left            -> (" + u.ToString("F3") + ", " + v.ToString("F3") + ")  (expected 288, 180)  " + Check(System.Math.Abs(u - 288) < 1e-3 && System.Math.Abs(v - 180) < 1e-3));
        CalibrationBuilder.Project(c2.camera, tx + 10 * r[0, 0] + r[0, 2], ty + 10 * r[1, 0] + r[1, 2], tz + 10 * r[2, 0] + r[2, 2], out u, out v, out d);
        sb.AppendLine("+ 1 m up                     -> (" + u.ToString("F3") + ", " + v.ToString("F3") + ")  (expected 320, 148)  " + Check(System.Math.Abs(u - 320) < 1e-3 && System.Math.Abs(v - 148) < 1e-3));

        UnityEngine.Debug.Log(sb.ToString());
    }

    [ContextMenu("Save calibration as Python test fixture")]
    void SaveFixture()
    {
        SensorConfig cfg = SensorConfigLoader.Load("default");
        CalibrationData c = CalibrationBuilder.Build(cfg, vehicle, egoBox);
        string text = CalibrationBuilder.ToJson(c, true);
        string root = System.IO.Directory.GetParent(ConfigPaths.ConfigDir()).FullName;
        string dir = System.IO.Path.Combine(root, "testing", "data", "calibration");
        System.IO.Directory.CreateDirectory(dir);
        System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "recorded_data.json"), text);
        UnityEngine.Debug.Log("Saved calibration " + c.calibrationId + " to " + dir);
    }
}