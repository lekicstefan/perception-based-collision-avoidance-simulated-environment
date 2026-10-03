using UnityEngine;

// Speed profile along a route: min(cruise, sqrt(aLat / |curvature|)), then a backward pass
// (comfortable deceleration, ends at vEnd) and a forward pass (comfortable acceleration).
public class SpeedProfile
{
    readonly float[] v;
    readonly float ds;

    public SpeedProfile(Route route, float cruise, float aLat, float aDecel, float aAccel,
                        float vStart = 0f, float vEnd = 0f, float step = 0.5f)
    {
        int n = Mathf.CeilToInt(route.Length / step) + 1;
        ds = route.Length / (n - 1);
        v = new float[n];
        for (int i = 0; i < n; i++)
        {
            float k = Mathf.Abs(route.Sample(i * ds).curvature);
            v[i] = Mathf.Min(cruise, Mathf.Sqrt(aLat / Mathf.Max(k, 1e-4f)));
        }
        v[n - 1] = Mathf.Min(v[n - 1], vEnd);
        for (int i = n - 2; i >= 0; i--)
            v[i] = Mathf.Min(v[i], Mathf.Sqrt(v[i + 1] * v[i + 1] + 2f * aDecel * ds));
        v[0] = Mathf.Min(v[0], vStart);
        for (int i = 1; i < n; i++)
            v[i] = Mathf.Min(v[i], Mathf.Sqrt(v[i - 1] * v[i - 1] + 2f * aAccel * ds));
    }

    // Profile speed and feed-forward acceleration (v dv/ds) at arc length s.
    public void At(float s, out float speed, out float accel)
    {
        float f = Mathf.Clamp(s / ds, 0f, v.Length - 1.0001f);
        int i = (int)f;
        float a = f - i;
        speed = Mathf.Lerp(v[i], v[i + 1], a);
        accel = (v[i + 1] * v[i + 1] - v[i] * v[i]) / (2f * ds);
    }
}