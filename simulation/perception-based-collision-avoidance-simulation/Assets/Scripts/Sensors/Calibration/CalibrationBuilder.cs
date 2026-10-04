using Newtonsoft.Json;
using UnityEngine;

public static class CalibrationBuilder
{
    const double D = System.Math.PI / 180.0;

    static double[,] Mul(double[,] a, double[,] b)
    {
        var r = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++) r[i, j] += a[i, k] * b[k, j];
        return r;
    }

    static double[,] Transpose(double[,] a)
    {
        var r = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) r[i, j] = a[j, i];
        return r;
    }

    // vehicle_from_sensor: Rz(yaw) * Ry(-pitch) * Rx(roll)  (yaw left+, pitch up+, roll right-hand about forward)
    public static double[,] RotationFromMount(Mount m)
    {
        double yw = m.yawDeg * D, p = -m.pitchDeg * D, r = m.rollDeg * D;
        double cz = System.Math.Cos(yw), sz = System.Math.Sin(yw);
        double cy = System.Math.Cos(p), sy = System.Math.Sin(p);
        double cx = System.Math.Cos(r), sx = System.Math.Sin(r);
        var rz = new double[,] { { cz, -sz, 0 }, { sz, cz, 0 }, { 0, 0, 1 } };
        var ry = new double[,] { { cy, 0, sy }, { 0, 1, 0 }, { -sy, 0, cy } };
        var rx = new double[,] { { 1, 0, 0 }, { 0, cx, -sx }, { 0, sx, cx } };
        return Mul(Mul(rz, ry), rx);
    }

    static uint Fnv(string s)
    {
        unchecked
        {
            uint h = 2166136261u;
            foreach (char ch in s) { h ^= ch; h *= 16777619u; }
            return h;
        }
    }

    public static CalibrationData Build(SensorConfig cfg, Vehicle vehicle, BoxCollider egoBox)
    {
        var c = new CalibrationData();

        // ego geometry from the collision box, in the vehicle frame (Unity: x right, y up, z forward)
        Vector3 centre = vehicle.transform.InverseTransformPoint(egoBox.transform.TransformPoint(egoBox.center));
        Vector3 sc = egoBox.transform.lossyScale;
        float length = egoBox.size.z * sc.z;
        c.ego = new EgoCalibration
        {
            wheelbase = vehicle.wheelbase,
            length = length,
            width = egoBox.size.x * sc.x,
            height = egoBox.size.y * sc.y,
            rearOverhang = -(centre.z - 0.5f * length),
            frontOverhang = centre.z + 0.5f * length - vehicle.wheelbase
        };

        // camera: optical frame from the body frame: x_opt = -y, y_opt = -z, z_opt = x
        CameraConfig cam = cfg.camera;
        double[,] rc = RotationFromMount(cam.mount);
        var o = new double[,] { { 0, -1, 0 }, { 0, 0, -1 }, { 1, 0, 0 } };
        double[,] ropt = Mul(o, Transpose(rc));
        double[] tm = { cam.mount.x, cam.mount.y, cam.mount.z };
        var ot = new OpticalTransform();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++) ot.R[i * 3 + j] = ropt[i, j];
            ot.t[i] = -(ropt[i, 0] * tm[0] + ropt[i, 1] * tm[1] + ropt[i, 2] * tm[2]);
        }
        c.camera = new CameraCalibration
        {
            width = cam.width,
            height = cam.height,
            rateHz = cam.rateHz,
            format = cam.format,
            fx = cam.FocalPx,
            fy = cam.FocalPx,
            cx = cam.Cx,
            cy = cam.Cy,
            horizontalFovDeg = cam.horizontalFovDeg,
            verticalFovDeg = cam.VerticalFovDeg,
            mount = cam.mount,
            opticalFromVehicle = ot
        };

        LidarConfig l = cfg.lidar;
        var az = new float[l.Columns];
        for (int i = 0; i < az.Length; i++) az[i] = l.AzimuthDeg(i);
        c.lidar = new LidarCalibration
        {
            rows = l.Rows,
            cols = l.Columns,
            rateHz = l.rateHz,
            maxRangeM = l.maxRangeM,
            mount = l.mount,
            elevationsDeg = (float[])l.ElevationsDeg.Clone(),
            azimuthsDeg = az
        };
        c.poseRateHz = cfg.pose.rateHz;

        c.calibrationId = "";
        c.calibrationId = Fnv(ToJson(c, false)).ToString("X8");
        return c;
    }

    public static string ToJson(CalibrationData c, bool indented)
    {
        return JsonConvert.SerializeObject(c, indented ? Formatting.Indented : Formatting.None);
    }

    // vehicle-frame point to pixel coordinates (u, v) and depth along the optical axis
    public static void Project(CameraCalibration cam, double x, double y, double z, out double u, out double v, out double depth)
    {
        double[] r = cam.opticalFromVehicle.R, t = cam.opticalFromVehicle.t;
        double xo = r[0] * x + r[1] * y + r[2] * z + t[0];
        double yo = r[3] * x + r[4] * y + r[5] * z + t[1];
        depth = r[6] * x + r[7] * y + r[8] * z + t[2];
        u = cam.fx * xo / depth + cam.cx;
        v = cam.fy * yo / depth + cam.cy;
    }
}