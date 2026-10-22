using System.Collections.Generic;
using UnityEngine;

// Adds sidewalks with curbs and road markings along a route. Put it on the same GameObject as RouteRoadMesh (it copies the route file
// and the road width from there) or on any object at the origin and fill in routeFile and roadWidth.
// Built once at Awake, like the road itself. Both meshes are scenery for the camera and the lidar, nothing here is ground truth:
//   Sidewalks  one mesh with a MeshCollider on the Ground layer (the lidar sees the curb and the sidewalk, the ego car never collides with them)
//   Markings   centre dashes and edge lines, 2 cm above the road, no collider (the lidar cannot see them, the camera can)
//   Crossings  zebra crossings (list 'zebraCrossings'), like the markings: 2.5 cm above the road, no collider
// The materials are optional: the ones that the Environment Kit makes (Tools > Environment Kit) can be assigned here, otherwise grey ones are made.
[DefaultExecutionOrder(-5)]
public class RouteRoadDressing : MonoBehaviour
{
    [Header("Road (copied from RouteRoadMesh when one is on this object)")]
    public RouteRoadMesh roadMesh;
    public string routeFile = "routes/dev_road.json";
    public float roadWidth = 7f;
    public float centreOffsetM = 0f;             // copied from RouteRoadMesh: the road is this far to the left of the route

    [Header("Sidewalks")]
    public bool leftSidewalk = true;
    public bool rightSidewalk = true;
    public float sidewalkWidth = 3f;
    public float curbHeight = 0.12f;
    public int groundLayer = 3;                  // Ground
    public Material sidewalkMaterial;
    public Material curbMaterial;

    [Header("Markings")]
    public bool centreLine = true;
    public bool edgeLines = true;
    public float lineWidth = 0.15f;
    public float dashLengthM = 3f;
    public float dashGapM = 6f;
    public float edgeInsetM = 0.3f;              // distance of the edge lines from the road edge
    public Material centreLineMaterial;
    public Material edgeLineMaterial;

    [Header("Zebra crossings")]
    [Tooltip("One entry per crossing: x = distance along the route of its centre (m), y = its length along the route (m, 3 is normal). Put the conflict point of a crossing pedestrian in the middle.")]
    public List<Vector2> zebraCrossings = new List<Vector2>();
    public float zebraStripeWidthM = 0.5f;
    public float zebraGapM = 0.5f;
    public Material zebraMaterial;               // missing: the edge line material is used

    public float spacing = 1f;                   // section length along the route
    public float uvTileM = 2f;                   // texture tile size of the sidewalk, metres

    string status = "not built";

    void Awake()
    {
        try { Build(); }
        catch (System.Exception e)
        {
            status = "FAILED: " + e.Message;
            UnityEngine.Debug.LogError("RouteRoadDressing '" + name + "' failed: " + e);
        }
    }

    // ---- mesh helper -------------------------------------------------------------------------------------------------------------------

    class MeshData
    {
        public List<Vector3> v = new List<Vector3>();
        public List<Vector3> n = new List<Vector3>();
        public List<Vector2> uv = new List<Vector2>();
        public List<int> t = new List<int>();

        // quad (a0 a1 / b0 b1): a0 and a1 are consecutive points on one edge, b0 and b1 the matching points on the other.
        // The winding is chosen so that the front face looks along 'normal'.
        public void Quad(Vector3 a0, Vector3 a1, Vector3 b0, Vector3 b1, Vector2 ua0, Vector2 ua1, Vector2 ub0, Vector2 ub1, Vector3 normal)
        {
            int i = v.Count;
            v.Add(a0); v.Add(a1); v.Add(b0); v.Add(b1);
            uv.Add(ua0); uv.Add(ua1); uv.Add(ub0); uv.Add(ub1);
            for (int k = 0; k < 4; k++) n.Add(normal);
            // triangles (a0, a1, b0) and (b0, a1, b1); front normal of a triangle (p, q, r) is Cross(q - p, r - p)
            bool flip = Vector3.Dot(Vector3.Cross(a1 - a0, b0 - a0), normal) < 0f;
            if (!flip) { t.Add(i); t.Add(i + 1); t.Add(i + 2); t.Add(i + 2); t.Add(i + 1); t.Add(i + 3); }
            else { t.Add(i); t.Add(i + 2); t.Add(i + 1); t.Add(i + 2); t.Add(i + 3); t.Add(i + 1); }
        }

        public Mesh ToMesh(string meshName)
        {
            var m = new Mesh { name = meshName };
            if (v.Count > 65000) m.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            m.SetVertices(v);
            m.SetNormals(n);
            m.SetUVs(0, uv);
            m.SetTriangles(t, 0);
            m.RecalculateBounds();
            return m;
        }
    }

    static Material MakeMaterial(Color c)
    {
        Shader sh = Shader.Find("Standard");
        if (sh == null) sh = Shader.Find("Universal Render Pipeline/Lit");
        var m = new Material(sh) { color = c };
        if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 0f);
        return m;
    }

    // ---- build -------------------------------------------------------------------------------------------------------------------------

    void Build()
    {
        if (roadMesh == null) roadMesh = GetComponent<RouteRoadMesh>();
        if (roadMesh != null) { routeFile = roadMesh.routeFile; roadWidth = roadMesh.width; centreOffsetM = roadMesh.centreOffsetM; }
        if (transform.position != Vector3.zero || transform.rotation != Quaternion.identity || transform.lossyScale != Vector3.one)
            UnityEngine.Debug.LogWarning("RouteRoadDressing '" + name + "' is not at the origin with rotation 0 and scale 1: the dressing will be displaced");

        Route r = RouteIO.Build(RouteIO.Read(ConfigPaths.Resolve(routeFile)));
        int n = Mathf.CeilToInt(r.Length / Mathf.Max(0.1f, spacing)) + 1;
        float ds = r.Length / (n - 1);
        float half = 0.5f * roadWidth;

        // centre line points and the unit vector to the left of the route, per section
        var centre = new Vector3[n];
        var left = new Vector3[n];
        for (int i = 0; i < n; i++)
        {
            RouteSample s = r.Sample(i * ds);
            left[i] = new Vector3(-Mathf.Sin(s.heading), 0f, Mathf.Cos(s.heading));
            centre[i] = s.position + left[i] * centreOffsetM;
        }

        BuildSidewalks(centre, left, ds, half);
        BuildMarkings(r, half);
        BuildCrossings(r, half);
        status = "OK";
        UnityEngine.Debug.Log("ROAD DRESSING built from " + routeFile + ": road width " + roadWidth.ToString("F1") + " m, sidewalks " +
            (leftSidewalk ? "left " : "") + (rightSidewalk ? "right " : "") + sidewalkWidth.ToString("F1") + " m wide, curb " + curbHeight.ToString("F2") + " m");
    }

    void BuildSidewalks(Vector3[] centre, Vector3[] left, float ds, float half)
    {
        if (!leftSidewalk && !rightSidewalk) return;
        var top = new MeshData();       // sidewalk top and the outer face
        var curb = new MeshData();      // the face toward the road
        float tile = Mathf.Max(0.1f, uvTileM);
        int n = centre.Length;
        if (sidewalkMaterial == null) sidewalkMaterial = MakeMaterial(new Color(0.62f, 0.62f, 0.62f));
        if (curbMaterial == null) curbMaterial = MakeMaterial(new Color(0.78f, 0.78f, 0.78f));

        for (int side = 0; side < 2; side++)
        {
            if (side == 0 && !leftSidewalk) continue;
            if (side == 1 && !rightSidewalk) continue;
            float sign = side == 0 ? 1f : -1f;
            for (int i = 0; i + 1 < n; i++)
            {
                Vector3 l0 = left[i] * sign, l1 = left[i + 1] * sign;       // outward direction at both ends
                Vector3 c0 = centre[i], c1 = centre[i + 1];
                Vector3 roadA = c0 + l0 * half, roadB = c1 + l1 * half;
                Vector3 outA = c0 + l0 * (half + sidewalkWidth), outB = c1 + l1 * (half + sidewalkWidth);
                Vector3 h = Vector3.up * curbHeight;
                float s0 = i * ds / tile, s1 = (i + 1) * ds / tile;
                float w = sidewalkWidth / tile;
                Vector3 outward = (l0 + l1).normalized;

                // curb face: from the road surface up to the sidewalk, facing the road
                curb.Quad(roadA, roadB, roadA + h, roadB + h,
                    new Vector2(0f, s0), new Vector2(0f, s1), new Vector2(curbHeight / tile, s0), new Vector2(curbHeight / tile, s1), -outward);
                // sidewalk top
                top.Quad(roadA + h, roadB + h, outA + h, outB + h,
                    new Vector2(0f, s0), new Vector2(0f, s1), new Vector2(w, s0), new Vector2(w, s1), Vector3.up);
                // outer face (never seen from the road, but keeps the mesh closed to the ground)
                top.Quad(outA + h, outB + h, outA, outB,
                    new Vector2(0f, s0), new Vector2(0f, s1), new Vector2(curbHeight / tile, s0), new Vector2(curbHeight / tile, s1), outward);
            }
        }

        // one mesh, two sub-meshes (sidewalk, curb), one collider
        var all = new MeshData();
        all.v.AddRange(top.v); all.n.AddRange(top.n); all.uv.AddRange(top.uv); all.t.AddRange(top.t);
        int off = all.v.Count;
        all.v.AddRange(curb.v); all.n.AddRange(curb.n); all.uv.AddRange(curb.uv);
        var curbTris = new List<int>();
        foreach (int idx in curb.t) curbTris.Add(idx + off);

        var mesh = new Mesh { name = "Sidewalks" };
        if (all.v.Count > 65000) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(all.v);
        mesh.SetNormals(all.n);
        mesh.SetUVs(0, all.uv);
        mesh.subMeshCount = 2;
        mesh.SetTriangles(top.t, 0);
        mesh.SetTriangles(curbTris, 1);
        mesh.RecalculateBounds();

        var go = new GameObject("Sidewalks");
        go.transform.SetParent(transform, false);
        go.layer = groundLayer;
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        go.AddComponent<MeshRenderer>().sharedMaterials = new[] { sidewalkMaterial, curbMaterial };
        go.AddComponent<MeshCollider>().sharedMesh = mesh;
    }

    void BuildMarkings(Route r, float half)
    {
        if (!centreLine && !edgeLines) return;
        var dashes = new MeshData();
        var edges = new MeshData();
        float y = 0.02f;
        float hw = 0.5f * lineWidth;
        float step = Mathf.Max(0.25f, Mathf.Min(spacing, 1f));

        if (centreLine)
        {
            float period = dashLengthM + dashGapM;
            for (float s0 = 0f; s0 < r.Length; s0 += period)
                AddStrip(dashes, r, s0, Mathf.Min(s0 + dashLengthM, r.Length), centreOffsetM, hw, y, step);
        }
        if (edgeLines)
        {
            float off = half - edgeInsetM;
            AddStrip(edges, r, 0f, r.Length, centreOffsetM + off, hw, y, step);
            AddStrip(edges, r, 0f, r.Length, centreOffsetM - off, hw, y, step);
        }

        if (centreLineMaterial == null) centreLineMaterial = MakeMaterial(new Color(1f, 0.82f, 0f));
        if (edgeLineMaterial == null) edgeLineMaterial = MakeMaterial(new Color(0.85f, 0.85f, 0.85f));

        var go = new GameObject("Markings");
        go.transform.SetParent(transform, false);
        go.layer = 0;
        var mesh = new Mesh { name = "Markings" };
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        var v = new List<Vector3>(dashes.v);
        var nn = new List<Vector3>(dashes.n);
        var uv = new List<Vector2>(dashes.uv);
        int off2 = v.Count;
        v.AddRange(edges.v); nn.AddRange(edges.n); uv.AddRange(edges.uv);
        var edgeTris = new List<int>();
        foreach (int idx in edges.t) edgeTris.Add(idx + off2);
        mesh.SetVertices(v);
        mesh.SetNormals(nn);
        mesh.SetUVs(0, uv);
        mesh.subMeshCount = 2;
        mesh.SetTriangles(dashes.t, 0);
        mesh.SetTriangles(edgeTris, 1);
        mesh.RecalculateBounds();
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        go.AddComponent<MeshRenderer>().sharedMaterials = new[] { centreLineMaterial, edgeLineMaterial };
    }

    // zebra stripes parallel to the road, across its whole width (inside the edge lines), at every entry of zebraCrossings
    void BuildCrossings(Route r, float half)
    {
        if (zebraCrossings == null || zebraCrossings.Count == 0) return;
        var m = new MeshData();
        float pitch = zebraStripeWidthM + zebraGapM;
        float span = 2f * (half - edgeInsetM) - 0.2f;
        int n = Mathf.Max(1, Mathf.FloorToInt((span + zebraGapM) / pitch));
        float first = -0.5f * (n - 1) * pitch;
        float step = Mathf.Max(0.25f, Mathf.Min(spacing, 1f));
        foreach (Vector2 c in zebraCrossings)
        {
            float s0 = Mathf.Max(0f, c.x - 0.5f * c.y), s1 = Mathf.Min(r.Length, c.x + 0.5f * c.y);
            for (int k = 0; k < n; k++)
                AddStrip(m, r, s0, s1, centreOffsetM + first + k * pitch, 0.5f * zebraStripeWidthM, 0.025f, step);
        }
        if (zebraMaterial == null) zebraMaterial = edgeLineMaterial != null ? edgeLineMaterial : MakeMaterial(new Color(0.92f, 0.92f, 0.92f));
        var go = new GameObject("Crossings");
        go.transform.SetParent(transform, false);
        go.layer = 0;
        go.AddComponent<MeshFilter>().sharedMesh = m.ToMesh("Crossings");
        go.AddComponent<MeshRenderer>().sharedMaterial = zebraMaterial;
    }

    // a strip of the given half width, centred 'offset' metres to the left of the route, from distance s0 to s1
    static void AddStrip(MeshData m, Route r, float s0, float s1, float offset, float halfWidth, float y, float step)
    {
        if (s1 - s0 < 0.01f) return;
        int segs = Mathf.Max(1, Mathf.CeilToInt((s1 - s0) / step));
        float d = (s1 - s0) / segs;
        for (int k = 0; k < segs; k++)
        {
            Vector3 a0, a1, b0, b1;
            Ends(r, s0 + k * d, offset, halfWidth, y, out a0, out b0);
            Ends(r, s0 + (k + 1) * d, offset, halfWidth, y, out a1, out b1);
            m.Quad(a0, a1, b0, b1, new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(1f, 0f), new Vector2(1f, 1f), Vector3.up);
        }
    }

    static void Ends(Route r, float s, float offset, float halfWidth, float y, out Vector3 leftEnd, out Vector3 rightEnd)
    {
        RouteSample smp = r.Sample(s);
        Vector3 left = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading));
        Vector3 p = smp.position + left * offset;
        p.y += y;
        leftEnd = p + left * halfWidth;
        rightEnd = p - left * halfWidth;
    }

    void OnGUI()
    {
        if (status == "OK") return;
        GUI.color = Color.red;
        GUI.Label(new Rect(10, Screen.height - 96, 900, 22), "ROAD DRESSING " + status + " (see the Console)");
    }
}