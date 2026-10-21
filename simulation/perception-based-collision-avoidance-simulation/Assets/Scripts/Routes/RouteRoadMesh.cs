using UnityEngine;

// Builds a flat road ribbon from a route file. The GameObject must be at the origin with no rotation and scale 1.
// The mesh has texture coordinates in metres: u across the road and v along it, both divided by uvTileM, so a texture
// with one grid cell per tile looks the same on every road, whatever its length (see docs/scenery.md).
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
public class RouteRoadMesh : MonoBehaviour
{
    public string routeFile = "routes/dev_road.json";
    public float width = 7f;
    public float spacing = 1f;
    public float uvTileM = 4f;       // size of one texture tile in metres
    public float centreOffsetM = 0f; // moves the road to the LEFT of the route by this much (negative: right). 0 = the route is the centre line.
                                     // Half a lane width (width / 4) puts the route in the right-hand lane. RouteRoadDressing and RoadsideScenery follow it.

    string status = "not built";

    void Awake()
    {
        try { Build(); }
        catch (System.Exception e)
        {
            status = "FAILED: " + e.Message;
            UnityEngine.Debug.LogError("RouteRoadMesh '" + name + "' could not build '" + routeFile + "': " + e);
        }
    }

    void Build()
    {
        if (transform.position != Vector3.zero || transform.rotation != Quaternion.identity || transform.lossyScale != Vector3.one)
            UnityEngine.Debug.LogWarning("RouteRoadMesh '" + name + "' is not at the origin with rotation 0 and scale 1: the road will be displaced");

        Route r = RouteIO.Build(RouteIO.Read(ConfigPaths.Resolve(routeFile)));
        int n = Mathf.CeilToInt(r.Length / spacing) + 1;
        float ds = r.Length / (n - 1);
        float tile = Mathf.Max(0.1f, uvTileM);

        var verts = new Vector3[2 * n];
        var uvs = new Vector2[2 * n];
        for (int i = 0; i < n; i++)
        {
            RouteSample smp = r.Sample(i * ds);
            Vector3 l = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading));
            Vector3 c = smp.position + l * centreOffsetM;
            verts[2 * i] = c + l * (0.5f * width);
            verts[2 * i + 1] = c - l * (0.5f * width);
            uvs[2 * i] = new Vector2(0f, i * ds / tile);
            uvs[2 * i + 1] = new Vector2(width / tile, i * ds / tile);
        }
        var tris = new int[(n - 1) * 6];
        for (int i = 0; i < n - 1; i++)
        {
            int a = 2 * i, t = 6 * i;
            tris[t] = a; tris[t + 1] = a + 2; tris[t + 2] = a + 1;
            tris[t + 3] = a + 1; tris[t + 4] = a + 2; tris[t + 5] = a + 3;
        }

        var mesh = new Mesh { name = "RoadRibbon" };
        mesh.vertices = verts;
        mesh.uv = uvs;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        GetComponent<MeshFilter>().sharedMesh = mesh;
        GetComponent<MeshCollider>().sharedMesh = mesh;

        var mr = GetComponent<MeshRenderer>();
        if (mr.sharedMaterial == null)
        {
            Shader sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Standard");
            mr.sharedMaterial = new Material(sh) { color = new Color(0.25f, 0.25f, 0.27f) };
            UnityEngine.Debug.LogWarning("RouteRoadMesh '" + name + "' had no material, a grey one was created");
        }
        status = "OK";
        UnityEngine.Debug.Log("ROAD built from " + routeFile + ": " + n + " sections, length " + r.Length.ToString("F1") + " m, bounds centre " +
            mesh.bounds.center + " size " + mesh.bounds.size + ", layer " + LayerMask.LayerToName(gameObject.layer) +
            ", renderer enabled " + mr.enabled + ", object active " + gameObject.activeInHierarchy);
    }

    void OnGUI()
    {
        if (status == "OK") return;
        GUI.color = Color.red;
        GUI.Label(new Rect(10, Screen.height - 74, 900, 22), "ROAD " + status + " (see the Console)");
    }
}