using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class RouteAuthoring : MonoBehaviour
{
    public string routeName = "test_route";
    public float gizmoStep = 0.5f;

    List<Vector3> Waypoints()
    {
        var l = new List<Vector3>();
        foreach (Transform c in transform) l.Add(c.position);
        return l;
    }

    string FilePath()
    {
        return ConfigPaths.Resolve(Path.Combine("routes", routeName + ".json"));
    }

    void OnDrawGizmos()
    {
        var wp = Waypoints();
        Gizmos.color = Color.yellow;
        foreach (var w in wp) Gizmos.DrawSphere(w, 0.6f);
        if (wp.Count < 2) return;

        Route r = null;
        try { r = new Route(wp, gizmoStep, routeName); } catch (ArgumentException) { }
        if (r == null) // invalid, e.g. two waypoints closer than 0.5 m
        {
            Gizmos.color = Color.red;
            for (int i = 0; i + 1 < wp.Count; i++) Gizmos.DrawLine(wp[i], wp[i + 1]);
            return;
        }
        Gizmos.color = Color.cyan;
        Vector3 prev = r.Sample(0f).position;
        for (float s = gizmoStep; s < r.Length + gizmoStep; s += gizmoStep)
        {
            Vector3 cur = r.Sample(s).position;
            Gizmos.DrawLine(prev, cur);
            prev = cur;
        }
    }

    Transform AddWaypoint(Vector3 position)
    {
        var go = new GameObject($"WP_{transform.childCount:D3}");
        go.transform.SetParent(transform, false);
        go.transform.position = position;
        return go.transform;
    }

    [ContextMenu("Add waypoint")]
    void AddNext()
    {
        int n = transform.childCount;
        Vector3 p;
        if (n == 0) p = transform.position;
        else if (n == 1) p = transform.GetChild(0).position + Vector3.forward * 10f;
        else
        {
            Vector3 last = transform.GetChild(n - 1).position;
            p = last + (last - transform.GetChild(n - 2).position);
        }
        AddWaypoint(p);
    }

    [ContextMenu("Export route JSON")]
    void Export()
    {
        var pts = Waypoints();
        new Route(pts, 0.1f, routeName); // throws if the route is invalid
        var f = new RouteFile { name = routeName };
        foreach (var w in pts) f.waypoints.Add(new RouteWaypoint { x = w.x, z = w.z, elevation = w.y });
        RouteIO.Write(FilePath(), f);
        UnityEngine.Debug.Log("Route written to " + FilePath());
    }

    [ContextMenu("Import route JSON (replaces children)")]
    void Import()
    {
        var f = RouteIO.Read(FilePath());
        for (int i = transform.childCount - 1; i >= 0; i--) DestroyImmediate(transform.GetChild(i).gameObject);
        foreach (var w in f.waypoints) AddWaypoint(new Vector3(w.x, w.elevation, w.z));
        UnityEngine.Debug.Log($"Imported {f.waypoints.Count} waypoints from {FilePath()}");
    }

    [ContextMenu("Self-test (circle and ramp)")]
    void SelfTest()
    {
        // left-turning quarter circle, radius 50 m, starting at the origin heading +Z
        const float R = 50f;
        var c = new List<Vector3>();
        for (int d = 0; d <= 90; d += 10)
        {
            float phi = d * Mathf.Deg2Rad;
            c.Add(new Vector3(-R + R * Mathf.Cos(phi), 0f, R * Mathf.Sin(phi)));
        }
        var rc = new Route(c, 0.1f, "circle");
        float sum = 0f; int cnt = 0;
        for (float s = 0.25f * rc.Length; s <= 0.75f * rc.Length; s += 0.5f) { sum += rc.Sample(s).curvature; cnt++; }
        UnityEngine.Debug.Log($"CIRCLE length {rc.Length:F3} m (expect 78.53), mean curvature {sum / cnt:F5} 1/m (expect +0.02000), " +
            $"start heading {rc.Sample(0f).heading * Mathf.Rad2Deg:F1} deg, end heading {rc.Sample(rc.Length).heading * Mathf.Rad2Deg:F1} deg (expect about 95 and 175)");

        // straight ramp: 10 m rise over 100 m
        var ramp = new List<Vector3>();
        for (int z = 0; z <= 100; z += 20) ramp.Add(new Vector3(0f, 0.1f * z, z));
        var rr = new Route(ramp, 0.1f, "ramp");
        RouteSample m = rr.Sample(50f);
        UnityEngine.Debug.Log($"RAMP length {rr.Length:F3} m (expect 100), elevation at s=50 {m.position.y:F3} m (expect 5), " +
            $"grade {m.grade:F4} (expect 0.1), curvature {m.curvature:F5} (expect 0)");
    }
}