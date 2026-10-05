using System.IO;
using System.Text;
using UnityEngine;

public class SensorConfigTest : MonoBehaviour
{
    public string configName = "default";

    static bool Near(Vector3 a, Vector3 b) { return (a - b).magnitude < 1e-5f; }

    [ContextMenu("Report")]
    void Report()
    {
        var sb = new StringBuilder();
        SensorConfig c = SensorConfigLoader.Load(configName);
        CameraConfig cam = c.camera;
        LidarConfig l = c.lidar;

        sb.AppendLine("SENSOR CONFIG '" + c.name + "' (seed " + c.seed + ")");
        sb.AppendLine("camera: " + cam.width + "x" + cam.height + " @ " + cam.rateHz.ToString("F0") + " Hz, hFOV " +
            cam.horizontalFovDeg.ToString("F1") + ", vFOV " + cam.VerticalFovDeg.ToString("F2") + " deg, fx = fy = " +
            cam.FocalPx.ToString("F1") + " px, cx " + cam.Cx.ToString("F1") + ", cy " + cam.Cy.ToString("F1") + ", " +
            cam.format + " q" + cam.jpegQuality + ", mount (" + cam.mount.x + ", " + cam.mount.y + ", " + cam.mount.z + ")");
        sb.AppendLine("lidar: " + l.Rows + " rows x " + l.Columns + " columns @ " + l.rateHz.ToString("F0") + " Hz, hFOV " +
            l.horizontalFovDeg + ", resolution " + l.horizontalResolutionDeg + " deg, range " + l.maxRangeM + " m, mount (" +
            l.mount.x + ", " + l.mount.y + ", " + l.mount.z + ")");

        var el = new StringBuilder();
        foreach (float e in l.ElevationsDeg) el.Append(e.ToString("F3")).Append(' ');
        sb.AppendLine("elevations (deg, row 0 first): " + el);

        // central beams and vertical gap at 50 m
        int n = 0; float hi = -90f, lo = 90f;
        foreach (float e in l.ElevationsDeg) if (e >= -4.0001f && e <= 4.0001f) { n++; hi = Mathf.Max(hi, e); lo = Mathf.Min(lo, e); }
        float spacing = n > 1 ? (hi - lo) / (n - 1) : 0f;
        sb.AppendLine("central beams (-4..+4 deg): " + n + ", spacing " + spacing.ToString("F4") + " deg, vertical gap at 50 m " +
            (50f * Mathf.Tan(spacing * Mathf.Deg2Rad)).ToString("F3") + " m (spec: about 0.45 m)");

        // beams that hit flat ground within range
        int g = 0; float near = 1e9f, far = 0f, farDeg = 0f, nearDeg = 0f;
        foreach (float e in l.ElevationsDeg)
        {
            if (e >= 0f) continue;
            float d = l.mount.z / Mathf.Tan(-e * Mathf.Deg2Rad);
            if (d > l.maxRangeM) continue;
            g++;
            if (d < near) { near = d; nearDeg = e; }
            if (d > far) { far = d; farDeg = e; }
        }
        sb.AppendLine("beams hitting flat ground within range: " + g + ", nearest " + near.ToString("F2") + " m (" + nearDeg.ToString("F2") +
            " deg), farthest " + far.ToString("F2") + " m (" + farDeg.ToString("F2") + " deg)");

        // camera pixels per LiDAR cell
        float res = l.horizontalResolutionDeg * Mathf.Deg2Rad;
        float ppcCentre = cam.FocalPx * Mathf.Tan(res);
        float half = cam.horizontalFovDeg * 0.5f * Mathf.Deg2Rad;
        float ppcEdge = cam.FocalPx * (Mathf.Tan(half) - Mathf.Tan(half - res));
        sb.AppendLine("camera pixels per LiDAR cell: centre " + ppcCentre.ToString("F3") + ", edge " + ppcEdge.ToString("F3") +
            " (spec: about 1.1 and 2.2 at 640 px)");

        // frame conversions
        sb.AppendLine("frame conversion checks:");
        sb.AppendLine("  forward  -> Unity +Z : " + (Near(Frames.ToUnity(1, 0, 0), new Vector3(0, 0, 1)) ? "PASS" : "FAIL"));
        sb.AppendLine("  left     -> Unity -X : " + (Near(Frames.ToUnity(0, 1, 0), new Vector3(-1, 0, 0)) ? "PASS" : "FAIL"));
        sb.AppendLine("  up       -> Unity +Y : " + (Near(Frames.ToUnity(0, 0, 1), new Vector3(0, 1, 0)) ? "PASS" : "FAIL"));
        Vector3 yawLeft = Frames.MountRotationToUnity(90f, 0f, 0f) * Vector3.forward;
        sb.AppendLine("  yaw +90 turns forward to the left: " + (Near(yawLeft, new Vector3(-1, 0, 0)) ? "PASS" : "FAIL"));
        Vector3 pitchUp = Frames.MountRotationToUnity(0f, 10f, 0f) * Vector3.forward;
        sb.AppendLine("  pitch +10 tilts forward up       : " + (Near(pitchUp, new Vector3(0f, Mathf.Sin(10f * Mathf.Deg2Rad), Mathf.Cos(10f * Mathf.Deg2Rad))) ? "PASS" : "FAIL"));
        Vector3 rollRight = Frames.MountRotationToUnity(0f, 0f, 90f) * Vector3.up;
        sb.AppendLine("  roll +90 tilts up to the right   : " + (Near(rollRight, new Vector3(1, 0, 0)) ? "PASS" : "FAIL"));
        float fx, fy, fz;
        Frames.FromUnity(Frames.ToUnity(1.5f, -2f, 0.7f), out fx, out fy, out fz);
        sb.AppendLine("  round trip                       : " + (Mathf.Abs(fx - 1.5f) + Mathf.Abs(fy + 2f) + Mathf.Abs(fz - 0.7f) < 1e-5f ? "PASS" : "FAIL"));

        // default.json must equal the defaults in code
        string defaultPath = SensorConfigLoader.PathFor("default");
        if (File.Exists(defaultPath))
        {
            bool same = SensorConfigLoader.ToJson(SensorConfigLoader.Load("default")) == SensorConfigLoader.ToJson(SensorConfigLoader.Defaults());
            sb.AppendLine("default.json equals the code defaults: " + (same ? "yes" : "NO (update default.json or the defaults in code)"));
        }
        else sb.AppendLine("default.json not found at " + defaultPath);

        // every sensor config in the folder
        sb.AppendLine("configs/sensors:");
        string dir = Path.Combine(ConfigPaths.ConfigDir(), "sensors");
        foreach (string f in Directory.GetFiles(dir, "*.json"))
        {
            try
            {
                SensorConfig o = SensorConfigLoader.Load(f);
                sb.AppendLine("  " + Path.GetFileName(f) + ": lidar " + o.lidar.Rows + " x " + o.lidar.Columns + ", camera " + o.camera.width + "x" +
                    o.camera.height + " @ " + o.camera.rateHz.ToString("F0") + " Hz, pose noise " + (o.pose.noiseEnabled ? "on" : "off"));
            }
            catch (System.Exception ex) { sb.AppendLine("  " + Path.GetFileName(f) + ": ERROR " + ex.Message); }
        }
        UnityEngine.Debug.Log(sb.ToString());
    }
}