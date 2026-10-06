using UnityEngine;

// Builds a flat road ribbon from a route file (dev aid, also a start for graybox roads).
// The GameObject must be at the origin with no rotation and scale 1.
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
public class RouteRoadMesh : MonoBehaviour
{
    public string routeFile;
    public Vehicle vehicle;     // Used for copying the route file selected on the Vehicle object. Makes is so that the selection only needs to be made in one place.
    public float width = 7f;
    public float spacing = 1f;

    void Awake()
    {
        Route r = RouteIO.Build(RouteIO.Read(routeFile.Equals("") ? ConfigPaths.Resolve(vehicle.GetComponent<RouteDriver>().routeFile) : routeFile));
        int n = Mathf.CeilToInt(r.Length / spacing) + 1;
        float ds = r.Length / (n - 1);

        var verts = new Vector3[2 * n];
        for (int i = 0; i < n; i++)
        {
            RouteSample smp = r.Sample(i * ds);
            // left normal in the ground plane: (-sin psi, cos psi)
            Vector3 left = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading)) * (0.5f * width);
            verts[2 * i] = smp.position + left;      // left edge
            verts[2 * i + 1] = smp.position - left;  // right edge
        }

        // clockwise seen from above = front face up
        var tris = new int[(n - 1) * 6];
        for (int i = 0; i < n - 1; i++)
        {
            int a = 2 * i, t = 6 * i;
            tris[t] = a; tris[t + 1] = a + 2; tris[t + 2] = a + 1;
            tris[t + 3] = a + 1; tris[t + 4] = a + 2; tris[t + 5] = a + 3;
        }

        var mesh = new Mesh { name = "RoadRibbon" };
        mesh.vertices = verts;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        GetComponent<MeshFilter>().sharedMesh = mesh;
        GetComponent<MeshCollider>().sharedMesh = mesh;
    }
}