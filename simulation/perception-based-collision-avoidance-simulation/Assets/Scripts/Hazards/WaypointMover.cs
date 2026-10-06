using System.Collections.Generic;
using UnityEngine;

// Moves an object along waypoints at a constant speed (a minimal scripted mover; speed profiles and
// triggers come in step 8.3). The waypoints are the children of 'path', in Hierarchy order. Only x and z are
// used, the object keeps its height. At the start the object is placed on the first waypoint.
public class WaypointMover : MonoBehaviour
{
    public Transform path;
    public float speed = 1.4f;          // m/s along the path
    public float startDelay = 0f;       // s of simulation time before the object starts to move
    public bool faceDirection = true;   // turn to face the direction of travel
    public bool loop = false;

    readonly List<Vector3> pts = new List<Vector3>();
    int next;
    double elapsed;

    void Start()
    {
        foreach (Transform c in path) pts.Add(new Vector3(c.position.x, 0f, c.position.z));
        if (pts.Count < 2)
        {
            UnityEngine.Debug.LogWarning("WaypointMover '" + name + "' needs at least two waypoints");
            enabled = false;
            return;
        }
        Vector3 p = transform.position;
        transform.position = new Vector3(pts[0].x, p.y, pts[0].z);
        next = 1;
        Face(pts[1] - pts[0]);
    }

    void Face(Vector3 dir)
    {
        dir.y = 0f;
        if (faceDirection && dir.sqrMagnitude > 1e-8f)
            transform.rotation = Quaternion.LookRotation(dir.normalized, Vector3.up);
    }

    void FixedUpdate()
    {
        elapsed += Time.fixedDeltaTime;
        if (elapsed < startDelay) return;

        float budget = speed * Time.fixedDeltaTime;
        Vector3 pos = transform.position;
        while (budget > 1e-9f)
        {
            if (next >= pts.Count)
            {
                if (!loop) break;
                next = 1;
                pos = new Vector3(pts[0].x, pos.y, pts[0].z);
            }
            Vector3 target = new Vector3(pts[next].x, pos.y, pts[next].z);
            Vector3 delta = target - pos;
            float d = delta.magnitude;
            if (d <= budget)
            {
                pos = target;
                budget -= d;
                next++;
                if (next < pts.Count) Face(pts[next] - pts[next - 1]);
            }
            else
            {
                pos += delta / d * budget;
                budget = 0f;
                Face(delta);
            }
        }
        transform.position = pos;
    }

    void OnDrawGizmos()
    {
        if (path == null) return;
        Gizmos.color = Color.magenta;
        Transform prev = null;
        foreach (Transform c in path)
        {
            Gizmos.DrawSphere(c.position, 0.2f);
            if (prev != null) Gizmos.DrawLine(prev.position, c.position);
            prev = c;
        }
    }
}