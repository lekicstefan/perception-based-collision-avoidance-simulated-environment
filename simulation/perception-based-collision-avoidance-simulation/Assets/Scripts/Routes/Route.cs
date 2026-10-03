using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class RouteWaypoint
{
    public float x;          // Unity world X (m)
    public float z;          // Unity world Z (m)
    public float elevation;  // Unity world Y (m)
}

[Serializable]
public class RouteFile
{
    public int schema = 1;
    public string name;
    public List<RouteWaypoint> waypoints = new List<RouteWaypoint>();
}

public struct RouteSample
{
    public float s;           // plan-view arc length from the start (m)
    public Vector3 position;  // Unity world coordinates (x, elevation, z)
    public float heading;     // rad, CCW from +X seen from above (left turns increase it), continuous (not wrapped)
    public float curvature;   // 1/m, positive = left turn
    public float grade;       // d(elevation)/ds, rise over plan-view run
    public float YawDeg { get { return 90f - heading * Mathf.Rad2Deg; } } // Unity transform yaw
}

public class Route
{
    public string Name { get; private set; }
    public float Length { get; private set; }
    public float Step { get; private set; }

    Vector3[] pos;
    float[] heading, curvature, grade;
    int count;

    public Route(IList<Vector3> wp, float step = 0.1f, string name = "route")
    {
        if (wp == null || wp.Count < 2) throw new ArgumentException("A route needs at least 2 waypoints");
        for (int i = 0; i + 1 < wp.Count; i++)
            if (PlanDist(wp[i], wp[i + 1]) < 0.5f)
                throw new ArgumentException($"Waypoints {i} and {i + 1} are closer than 0.5 m in plan view");
        Name = name;
        int w = wp.Count;

        // phantom end points so the curve reaches the first and last waypoint
        var p = new Vector3[w + 2];
        p[0] = 2f * wp[0] - wp[1];
        for (int i = 0; i < w; i++) p[i + 1] = wp[i];
        p[w + 1] = 2f * wp[w - 1] - wp[w - 2];

        // centripetal knots on plan-view chord length
        var t = new float[w + 2];
        for (int i = 1; i < w + 2; i++) t[i] = t[i - 1] + Mathf.Sqrt(PlanDist(p[i], p[i - 1]));

        // dense evaluation (about 2 cm)
        var dpos = new List<Vector3>();
        for (int i = 0; i < w - 1; i++)
        {
            int sub = Mathf.Max(32, Mathf.CeilToInt(PlanDist(wp[i], wp[i + 1]) / 0.02f));
            for (int k = (i == 0 ? 0 : 1); k <= sub; k++)
            {
                float u = t[i + 1] + (t[i + 2] - t[i + 1]) * k / sub;
                dpos.Add(CatmullRom(p[i], p[i + 1], p[i + 2], p[i + 3],
                                    t[i], t[i + 1], t[i + 2], t[i + 3], u));
            }
        }

        // cumulative plan-view distance
        var cum = new double[dpos.Count];
        for (int j = 1; j < dpos.Count; j++) cum[j] = cum[j - 1] + PlanDist(dpos[j], dpos[j - 1]);
        double len = cum[cum.Length - 1];

        // resample at uniform arc-length spacing
        count = Mathf.Max(2, Mathf.CeilToInt((float)(len / step)) + 1);
        Step = (float)(len / (count - 1));
        Length = (float)len;
        pos = new Vector3[count];
        int jj = 0;
        for (int i = 0; i < count; i++)
        {
            double s = len * i / (count - 1);
            while (jj < cum.Length - 2 && cum[jj + 1] < s) jj++;
            double seg = cum[jj + 1] - cum[jj];
            float a = seg > 1e-9 ? (float)((s - cum[jj]) / seg) : 0f;
            pos[i] = Vector3.Lerp(dpos[jj], dpos[jj + 1], a);
        }

        // heading (central difference of positions, unwrapped)
        heading = new float[count];
        for (int i = 0; i < count; i++)
        {
            Vector3 a = pos[Mathf.Min(i + 1, count - 1)];
            Vector3 b = pos[Mathf.Max(i - 1, 0)];
            float h = Mathf.Atan2(a.z - b.z, a.x - b.x);
            if (i > 0)
            {
                while (h - heading[i - 1] > Mathf.PI) h -= 2f * Mathf.PI;
                while (h - heading[i - 1] < -Mathf.PI) h += 2f * Mathf.PI;
            }
            heading[i] = h;
        }

        // curvature and grade over a window of about +-0.5 m
        int win = Mathf.Max(1, Mathf.RoundToInt(0.5f / Step));
        var elev = new float[count];
        for (int i = 0; i < count; i++) elev[i] = pos[i].y;
        curvature = new float[count];
        grade = new float[count];
        for (int i = 0; i < count; i++)
        {
            curvature[i] = Deriv(heading, i, win);
            grade[i] = Deriv(elev, i, win);
        }
    }

    float Deriv(float[] f, int i, int k)
    {
        int lo = Mathf.Max(i - k, 0), hi = Mathf.Min(i + k, count - 1);
        return (f[hi] - f[lo]) / ((hi - lo) * Step);
    }

    public RouteSample Sample(float s)
    {
        s = Mathf.Clamp(s, 0f, Length);
        float f = s / Step;
        int i = Mathf.Min((int)f, count - 2);
        float a = f - i;
        return new RouteSample
        {
            s = s,
            position = Vector3.Lerp(pos[i], pos[i + 1], a),
            heading = Mathf.Lerp(heading[i], heading[i + 1], a),
            curvature = Mathf.Lerp(curvature[i], curvature[i + 1], a),
            grade = Mathf.Lerp(grade[i], grade[i + 1], a)
        };
    }

    // Plan-view closest point on the route, searched in [sHint - back, sHint + forward].
    public float FindClosestS(double x, double z, float sHint, float back = 3f, float forward = 15f)
    {
        int i0 = Mathf.Max(0, Mathf.FloorToInt((sHint - back) / Step));
        int i1 = Mathf.Min(count - 1, Mathf.CeilToInt((sHint + forward) / Step));
        int best = i0;
        double bestD = double.MaxValue;
        for (int i = i0; i <= i1; i++)
        {
            double dx = pos[i].x - x, dz = pos[i].z - z;
            double d = dx * dx + dz * dz;
            if (d < bestD) { bestD = d; best = i; }
        }

        // refine by projecting onto the two segments around the best table point
        double bestS = best * Step;
        bestD = double.MaxValue;
        for (int k = 0; k < 2; k++)
        {
            int a = best - 1 + k, b = a + 1;
            if (a < 0 || b > count - 1) continue;
            double abx = pos[b].x - pos[a].x, abz = pos[b].z - pos[a].z;
            double apx = x - pos[a].x, apz = z - pos[a].z;
            double t = (apx * abx + apz * abz) / (abx * abx + abz * abz);
            t = System.Math.Max(0.0, System.Math.Min(1.0, t));
            double qx = pos[a].x + t * abx - x, qz = pos[a].z + t * abz - z;
            double d = qx * qx + qz * qz;
            if (d < bestD) { bestD = d; bestS = (a + t) * Step; }
        }
        return (float)bestS;
    }

    static float PlanDist(Vector3 a, Vector3 b)
    {
        float dx = a.x - b.x, dz = a.z - b.z;
        return Mathf.Sqrt(dx * dx + dz * dz);
    }

    static Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3,
                              float t0, float t1, float t2, float t3, float t)
    {
        Vector3 a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1;
        Vector3 a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2;
        Vector3 a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3;
        Vector3 b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2;
        Vector3 b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3;
        return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2;
    }
}

public static class RouteIO
{
    public static RouteFile Read(string path)
    {
        return JsonUtility.FromJson<RouteFile>(File.ReadAllText(path));
    }

    public static void Write(string path, RouteFile f)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, JsonUtility.ToJson(f, true));
    }

    public static Route Build(RouteFile f, float step = 0.1f)
    {
        var pts = new List<Vector3>();
        foreach (var w in f.waypoints) pts.Add(new Vector3(w.x, w.elevation, w.z));
        return new Route(pts, step, f.name);
    }
}