using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Generates the objects described by a RoadsideScenery component (docs/scenery.md). Editor only.
public static class RoadsideSceneryBuilder
{
    const float CurbHeight = 0.12f;

    [MenuItem("Tools/Environment Kit/Generate roadside scenery for selected")]
    public static void GenerateSelected()
    {
        RoadsideScenery cfg = Selected();
        if (cfg == null) return;
        Generate(cfg);
    }

    [MenuItem("Tools/Environment Kit/Clear generated scenery of selected")]
    public static void ClearSelected()
    {
        RoadsideScenery cfg = Selected();
        if (cfg == null) return;
        Clear(cfg);
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
    }

    // The component on the selected object, on its parent or child, or the only one in the open scene.
    static RoadsideScenery Selected()
    {
        GameObject sel = Selection.activeGameObject;
        RoadsideScenery cfg = null;
        if (sel != null)
        {
            cfg = sel.GetComponentInParent<RoadsideScenery>();
            if (cfg == null) cfg = sel.GetComponentInChildren<RoadsideScenery>();
        }
        if (cfg != null) return cfg;

        RoadsideScenery[] all = UnityEngine.Object.FindObjectsByType<RoadsideScenery>(FindObjectsSortMode.None);
        if (all.Length == 1) return all[0];

        string selName = sel == null ? "nothing" : "'" + sel.name + "'" + (EditorUtility.IsPersistent(sel) ? " (an asset in the Project window, not an object of the open scene)" : "");
        EditorUtility.DisplayDialog("Environment Kit",
            "No RoadsideScenery component found.\n\nSelected: " + selName + "\nRoadsideScenery components in the open scene '" + SceneManager.GetActiveScene().name + "': " + all.Length +
            "\n\nClick the road object in the Hierarchy of the open scene (the one that has the component) and try again.", "OK");
        return null;
    }

    static void Clear(RoadsideScenery cfg)
    {
        Transform old = cfg.transform.Find("Generated");
        if (old != null) Undo.DestroyObjectImmediate(old.gameObject);
    }

    // prefabs of the kit with the given Hazard label and name prefix
    static List<GameObject> Load(string label, string prefix, List<string> exclude = null)
    {
        var list = new List<GameObject>();
        foreach (string guid in AssetDatabase.FindAssets("t:Prefab", new[] { EnvironmentKit.PreDir }))
        {
            var p = AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(guid));
            var h = p != null ? p.GetComponent<Hazard>() : null;
            if (h == null || h.label != label) continue;
            if (!string.IsNullOrEmpty(prefix) && !p.name.StartsWith(prefix, System.StringComparison.Ordinal)) continue;
            if (exclude != null && exclude.Contains(p.name)) continue;
            list.Add(p);
        }
        list.Sort((a, b) => string.CompareOrdinal(a.name, b.name));      // a stable order, so the same seed gives the same scenery
        return list;
    }

    static bool Cleared(RoadsideScenery cfg, float s0, float s1)
    {
        foreach (Vector2 r in cfg.keepClearM)
            if (s1 > r.x && s0 < r.y) return false;
        return true;
    }

    static GameObject Place(GameObject prefab, Vector3 pos, Quaternion rot, Transform parent)
    {
        var g = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        g.transform.SetParent(parent, false);
        g.transform.position = pos;
        g.transform.rotation = rot;
        return g;
    }

    public static void Generate(RoadsideScenery cfg)
    {
        if (cfg.roadMesh == null) cfg.roadMesh = cfg.GetComponent<RouteRoadMesh>();
        if (cfg.roadMesh != null) { cfg.routeFile = cfg.roadMesh.routeFile; cfg.roadWidth = cfg.roadMesh.width; cfg.centreOffsetM = cfg.roadMesh.centreOffsetM; }
        var dressing = cfg.GetComponent<RouteRoadDressing>();
        if (dressing != null) cfg.sidewalkWidth = dressing.sidewalkWidth;
        float c0 = cfg.centreOffsetM;                                  // the road centre is this far to the left of the route
        if (cfg.transform.position != Vector3.zero || cfg.transform.rotation != Quaternion.identity || cfg.transform.lossyScale != Vector3.one)
            UnityEngine.Debug.LogWarning("RoadsideScenery '" + cfg.name + "' is not at the origin with rotation 0 and scale 1: the scenery will be displaced against the road");

        Route r = RouteIO.Build(RouteIO.Read(ConfigPaths.Resolve(cfg.routeFile)));
        float end = cfg.endM > 0f ? Mathf.Min(cfg.endM, r.Length) : r.Length;
        float half = 0.5f * cfg.roadWidth;
        float verge = half + cfg.sidewalkWidth;                       // lateral distance of the outer sidewalk edge

        Clear(cfg);
        var container = new GameObject("Generated");
        Undo.RegisterCreatedObjectUndo(container, "Generate roadside scenery");
        container.transform.SetParent(cfg.transform, false);
        Transform parent = container.transform;

        var buildings = Load("building", cfg.buildingFilter, cfg.buildingExclude);
        var trees = Load("tree", cfg.treeFilter);
        var lampMatches = Load("lamp_post", "");
        GameObject lamp = lampMatches.Find(p => p.name == cfg.lampPrefab);
        var props = new List<GameObject>();
        foreach (string l in new[] { "trash_bin", "hydrant", "bench", "mail_box" }) props.AddRange(Load(l, "SP_"));
        int nB = 0, nT = 0, nL = 0, nP = 0;

        // buildings
        for (int side = 0; side < 2; side++)
        {
            if (buildings.Count == 0) break;
            if (side == 0 && !cfg.buildingsLeft) continue;
            if (side == 1 && !cfg.buildingsRight) continue;
            float sign = side == 0 ? 1f : -1f;
            var rng = new System.Random(cfg.seed * 31 + side);
            bool swap = Mathf.Abs(Mathf.RoundToInt(cfg.buildingFrontYaw / 90f)) % 2 == 1;
            float s = cfg.startM;
            while (s < end)
            {
                GameObject pf = buildings[rng.Next(buildings.Count)];
                Vector3 size = pf.GetComponent<BoxCollider>().size;
                float along = swap ? size.z : size.x, depth = swap ? size.x : size.z;
                if (s + along > end) break;
                float sc = s + along * 0.5f;
                if (Cleared(cfg, s, s + along))
                {
                    RouteSample smp = r.Sample(sc);
                    Vector3 left = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading));
                    Vector3 pos = smp.position + left * (c0 + sign * (verge + cfg.buildingSetbackM + depth * 0.5f));
                    pos.y = 0f;
                    Quaternion rot = Quaternion.LookRotation(-left * sign, Vector3.up) * Quaternion.Euler(0f, cfg.buildingFrontYaw, 0f);
                    Place(pf, pos, rot, parent);
                    nB++;
                }
                s += along + Mathf.Lerp(cfg.buildingGapMin, cfg.buildingGapMax, (float)rng.NextDouble());
            }
        }

        // trees (on the sidewalk near the curb)
        for (int side = 0; side < 2; side++)
        {
            if (trees.Count == 0) break;
            if (side == 0 && !cfg.treesLeft) continue;
            if (side == 1 && !cfg.treesRight) continue;
            float sign = side == 0 ? 1f : -1f;
            var rng = new System.Random(cfg.seed * 31 + 7 + side);
            float s = cfg.startM + (float)rng.NextDouble() * cfg.treeSpacingM;
            while (s < end)
            {
                if (Cleared(cfg, s - 1f, s + 1f))
                {
                    RouteSample smp = r.Sample(s);
                    Vector3 left = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading));
                    Vector3 pos = smp.position + left * (c0 + sign * (half + cfg.treeCurbOffsetM + (float)rng.NextDouble() * 0.4f));
                    pos.y = CurbHeight;
                    Place(trees[rng.Next(trees.Count)], pos, Quaternion.Euler(0f, (float)rng.NextDouble() * 360f, 0f), parent);
                    nT++;
                }
                s += Mathf.Max(3f, cfg.treeSpacingM + ((float)rng.NextDouble() * 2f - 1f) * cfg.treeJitterM);
            }
        }

        // street lamps, alternating sides
        if (lamp != null && cfg.lampSpacingM > 1f)
        {
            int k = 0;
            for (float s = cfg.startM + cfg.lampSpacingM * 0.5f; s < end; s += cfg.lampSpacingM, k++)
            {
                if (!Cleared(cfg, s - 1f, s + 1f)) continue;
                float sign = (k % 2 == 0) ? 1f : -1f;
                RouteSample smp = r.Sample(s);
                Vector3 left = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading));
                Vector3 pos = smp.position + left * (c0 + sign * (half + cfg.lampOffsetM));
                pos.y = CurbHeight;
                Place(lamp, pos, Quaternion.LookRotation(-left * sign, Vector3.up) * Quaternion.Euler(0f, cfg.lampFrontYaw, 0f), parent);
                nL++;
            }
        }
        else if (lamp == null) UnityEngine.Debug.LogWarning("RoadsideScenery: lamp prefab '" + cfg.lampPrefab + "' not found in " + EnvironmentKit.PreDir);

        // sidewalk props
        if (props.Count > 0)
        {
            var rng = new System.Random(cfg.seed * 31 + 13);
            int n = Mathf.RoundToInt(cfg.propsPer100M * (end - cfg.startM) / 100f);
            for (int i = 0; i < n; i++)
            {
                float s = Mathf.Lerp(cfg.startM, end, (float)rng.NextDouble());
                float sign = rng.Next(2) == 0 ? 1f : -1f;
                if (!Cleared(cfg, s - 1f, s + 1f)) continue;
                RouteSample smp = r.Sample(s);
                Vector3 left = new Vector3(-Mathf.Sin(smp.heading), 0f, Mathf.Cos(smp.heading));
                Vector3 pos = smp.position + left * (c0 + sign * (half + cfg.sidewalkWidth * 0.7f));
                pos.y = CurbHeight;
                Place(props[rng.Next(props.Count)], pos, Quaternion.LookRotation(-left * sign, Vector3.up) * Quaternion.Euler(0f, ((float)rng.NextDouble() - 0.5f) * 20f, 0f), parent);
                nP++;
            }
        }

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        UnityEngine.Debug.Log("ROADSIDE SCENERY generated along " + cfg.routeFile + ": " + nB + " buildings, " + nT + " trees, " + nL + " lamps, " + nP + " props");
    }
}