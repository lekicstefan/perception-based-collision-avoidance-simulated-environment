using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// The Environment Kit (docs/scenery.md). It turns the imported asset packs into ready-to-use scenery prefabs and makes the road and ground materials.
//
//   Tools > Environment Kit > Build kit                          materials, textures and one wrapper prefab per catalogue item in Assets/Environment
//   Tools > Environment Kit > Apply look to the open scene       sky, ambient light, directional light
//   Tools > Environment Kit > Add road dressing to selected road sidewalks and markings (RouteRoadDressing) with the kit materials; asphalt on the road
//   Tools > Environment Kit > Create showcase scene              every prefab in a row layout, to check size and facing
//
// A WRAPPER prefab is the pack model (a plain copy, its own colliders removed) under a root with:
//   * the pivot on the ground (centre of the footprint, or the base of poles, trees and people),
//   * ONE root collider fitted to the model (box, or an upright capsule for people, trees and poles),
//   * a Hazard component with a label, on the Hazard layer (ground truth and collisions, see docs/scenery.md).
// Run it again at any time: assets are updated in place. The catalogue below is the place to change scale, facing, labels and colliders.
public static class EnvironmentKit
{
    const string Root = "Assets/Environment";
    const string TexDir = Root + "/Textures";
    const string MatDir = Root + "/Materials";
    public const string PreDir = Root + "/Prefabs";
    const string ShowcasePath = "Assets/Scenes/EnvironmentShowcase.unity";
    const int LayerHazard = 7;
    const int LayerGround = 3;
    const int TexSize = 256;

    // ---- the catalogue -----------------------------------------------------------------------------------------------------------------

    const string Packs = "Assets/Asset Packs/";
    const string PD = Packs + "Pandazole_Ultimate_Pack/Pandazole City Town Pack/Prefabs/";
    const string SP = Packs + "SimplePoly - Town Pack/Prefabs/";
    const string KN = Packs + "KenneyCarKit/FBX format/";
    const string Banana = Packs + "Banana Yellow Games/Characters/Banana Man/Banana Man.fbx";

    // The Kenney cars are about 2.6 m long as imported. The ego car in the scenes is shown larger, so the other vehicles get the same factor.
    // Check it against the scale of the ego car's model object and change it here if they differ.
    const float KenneyScale = 1.7f;

    class Item
    {
        public string name, source, label;
        public float scale = 1f, yaw = 0f;      // yaw: turns the model so that its front is +Z
        public bool capsule;                    // upright capsule (people, trees, poles) instead of a box; the pivot stays where the model has it
        public float radius = 0.3f, heightFrac = 1f;
        public float patch;                     // diameter of a flat green disc around the foot (trees), 0 = none
        public bool aPose, yellow;              // people: arms down in an A pose, crash-test yellow body
        public bool dolly;                      // a small wheeled base and a pole under the model (the people), so that it looks like it rolls
        public float fitLength;                 // > 0: the scale is chosen so that the longest side (x or z) of the model is this long, in metres
        public bool animal;                     // pack models: scripts and physics components removed, the pack's materials re-made as Standard materials
    }

    static readonly List<Item> Catalog = new List<Item>();

    static void Box(string name, string source, string label, float scale = 1f, float yaw = 0f)
    { Catalog.Add(new Item { name = name, source = source, label = label, scale = scale, yaw = yaw }); }

    // an animal of the ithappy pack: 'find' is "find:<folder in Asset Packs>:<part of the file name>", the file is looked up by name
    static void Animal(string name, string find, string label, float lengthM)
    { Catalog.Add(new Item { name = name, source = find, label = label, fitLength = lengthM, animal = true }); }

    static void Cap(string name, string source, string label, float radius, float heightFrac = 1f, float scale = 1f, float yaw = 0f, bool dolly = false, float patch = 0f, bool person = false)
    { Catalog.Add(new Item { name = name, source = source, label = label, capsule = true, radius = radius, heightFrac = heightFrac, scale = scale, yaw = yaw, dolly = dolly, patch = patch, aPose = person, yellow = person }); }

    static void BuildCatalog()
    {
        Catalog.Clear();

        // people: the dummy is 1.9 m tall as imported
        Cap("Pedestrian_Adult", Banana, "pedestrian_adult", 0.30f, 1f, 0.92f, 0f, true, 0f, true);
        Cap("Pedestrian_Child", Banana, "pedestrian_child", 0.20f, 1f, 0.58f, 0f, true, 0f, true);

        // animals (ithappy Animals_FREE): the cat is scaled to 0.5 m long
        Animal("Animal_Cat", "find:ithappy:kitty", "cat", 0.5f);

        // vehicles (Kenney Car Kit, CC0, the same family as the ego car)
        foreach (string v in new[] { "sedan:car", "sedan-sports:car", "suv:car", "suv-luxury:car", "hatchback-sports:car", "taxi:car", "police:car",
                                     "van:van", "delivery:van", "ambulance:van", "truck:truck", "firetruck:truck", "garbage-truck:truck" })
        {
            string[] p = v.Split(':');
            Box("KN_" + p[0], KN + p[0] + ".fbx", p[1], KenneyScale);
        }

        // buildings
        foreach (string k in new[] { "CommercialBuilding:4", "CompanyBuilding:4", "Motel:5", "ResidentBuilding:6" })
        {
            string[] p = k.Split(':');
            for (int i = 1; i <= int.Parse(p[1]); i++)
                Box("PD_" + p[0] + "_0" + i, PD + "Env_" + p[0] + "_0" + i + ".prefab", "building");
        }
        for (int i = 1; i <= 35; i++)
            Box("SP_building_" + i.ToString("00"), SP + "Buildings/sptp_building_" + i.ToString("00") + ".prefab", "building");

        // trees: the collider is the trunk only, the crown is a picture (the lidar goes through it)
        for (int i = 1; i <= 8; i++)
            Cap("PD_Tree_0" + i, PD + "Prop_Tree_0" + i + ".prefab", "tree", 0.20f, 0.45f, 1f, 0f, false, 1.8f);
        for (int i = 1; i <= 2; i++)
            Cap("SP_tree_0" + i, SP + "Nature/sptp_tree_0" + i + ".prefab", "tree", 0.20f, 0.45f, 1f, 0f, false, 1.8f);
        for (int i = 1; i <= 2; i++)
        {
            Box("PD_Bush_0" + i, PD + "Prop_Bush_0" + i + ".prefab", "bush");
            Box("SP_bush_0" + i, SP + "Nature/sptp_bush_0" + i + ".prefab", "bush");
            Box("SP_stone_0" + i, SP + "Nature/sptp_stone_0" + i + ".prefab", "stone");
        }

        // street furniture
        for (int i = 1; i <= 3; i++) Box("PD_RoadCone_0" + i, PD + "Prop_RoadCone_0" + i + ".prefab", "traffic_cone");
        for (int i = 1; i <= 2; i++) Box("PD_TrashCan_0" + i, PD + "Prop_CTPTrashCan_0" + i + ".prefab", "trash_bin");
        Box("PD_OilBarrel_01", PD + "Prop_OilBerrel_01.prefab", "barrel");
        for (int i = 1; i <= 3; i++) Box("PD_TrashBag_0" + i, PD + "Prop_TrashBag_0" + i + ".prefab", "trash_bag");
        for (int i = 1; i <= 3; i++) Box("PD_ElectricCabinet_0" + i, PD + "Prop_ElectracityCabinet_0" + i + ".prefab", "cabinet");
        foreach (string s in new[] { "Stop", "Speed_30", "Speed_50", "NoParking", "NoEntery", "Footpath_01", "TrafficSignals", "Cross" })
            Cap("PD_Sign_" + s, PD + "Prop_StreetSign_" + s + ".prefab", "sign", 0.12f);
        Box("SP_bench_01", SP + "Props/sptp_bench_01.prefab", "bench");
        for (int i = 1; i <= 2; i++) Box("SP_bin_0" + i, SP + "Props/sptp_bin_0" + i + ".prefab", "trash_bin");
        Box("SP_dumpster_01", SP + "Props/sptp_dumpster_01.prefab", "dumpster");
        Box("SP_fire_hydrant_01", SP + "Props/sptp_fire_hydrant_01.prefab", "hydrant");
        Box("SP_mail_box_01", SP + "Props/sptp_mail_box_01.prefab", "mail_box");
        Box("SP_electrical_panel_01", SP + "Props/sptp_electrical_panel_01.prefab", "cabinet");
        for (int i = 1; i <= 2; i++) Cap("SP_street_light_0" + i, SP + "Props/sptp_street_light_0" + i + ".prefab", "lamp_post", 0.15f);
        Cap("SP_traffic_light_01", SP + "Props/sptp_traffic_light_01.prefab", "traffic_light", 0.15f);
        Cap("SP_billboard_03", SP + "Props/sptp_billboard_03.prefab", "billboard", 0.25f);
    }

    // ---- menu --------------------------------------------------------------------------------------------------------------------------

    // Existing wrapper prefabs keep the rotation and scale of their "Visual" child, so a fix made by hand in a prefab survives Build kit
    // (the pivot and the collider are fitted again). The second menu item ignores them and starts from the catalogue.
    static bool resetOrientation;

    [MenuItem("Tools/Environment Kit/Build kit")]
    public static void BuildKit() { resetOrientation = false; BuildKitInternal(); }

    [MenuItem("Tools/Environment Kit/Build kit (reset orientation and scale)")]
    public static void BuildKitReset() { resetOrientation = true; BuildKitInternal(); resetOrientation = false; }

    static void BuildKitInternal()
    {
        BuildMaterials();
        BuildGround();
        BuildCatalog();
        int ok = 0, missing = 0;
        var names = new List<string>();
        foreach (Item it in Catalog)
        {
            if (Wrap(it)) { ok++; names.Add(it.name); }
            else missing++;
        }
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        UnityEngine.Debug.Log("ENVIRONMENT KIT built: " + ok + " prefabs in " + PreDir + (missing > 0 ? ", " + missing + " sources not found (see the warnings above)" : ""));
    }

    [MenuItem("Tools/Environment Kit/Apply look to the open scene")]
    public static void ApplyLook()
    {
        EnsureKit();
        RenderSettings.skybox = AssetDatabase.LoadAssetAtPath<Material>(MatDir + "/Sky_Day.mat");
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Skybox;
        RenderSettings.ambientIntensity = 1f;

        Light sun = null;
        foreach (Light l in UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            if (l.type == LightType.Directional) { sun = l; break; }
        if (sun == null)
        {
            var go = new GameObject("Directional Light");
            sun = go.AddComponent<Light>();
            sun.type = LightType.Directional;
        }
        sun.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        sun.color = new Color(1f, 0.96f, 0.9f);
        sun.intensity = 1.1f;
        sun.shadows = LightShadows.Soft;
        RenderSettings.sun = sun;
        DynamicGI.UpdateEnvironment();
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        UnityEngine.Debug.Log("ENVIRONMENT KIT: look applied to scene '" + SceneManager.GetActiveScene().name + "'");
    }

    [MenuItem("Tools/Environment Kit/Add road dressing to selected road")]
    public static void AddRoadDressing()
    {
        GameObject sel = Selection.activeGameObject;
        RouteRoadMesh road = sel != null ? sel.GetComponent<RouteRoadMesh>() : null;
        if (road == null)
        {
            EditorUtility.DisplayDialog("Environment Kit", "Select the GameObject that has the RouteRoadMesh component (the road) in the Hierarchy first.", "OK");
            return;
        }
        DressRoad(sel);
        UnityEngine.Debug.Log("ENVIRONMENT KIT: road dressing added to '" + sel.name + "' (sidewalks and markings appear when the scene plays)");
    }

    // Adds (or updates) RouteRoadDressing on a road object and gives the road and the dressing the kit materials. Also used by the scenario scene builder.
    public static RouteRoadDressing DressRoad(GameObject roadObject)
    {
        RouteRoadMesh road = roadObject.GetComponent<RouteRoadMesh>();
        EnsureKit();
        var dressing = roadObject.GetComponent<RouteRoadDressing>();
        if (dressing == null) dressing = Undo.AddComponent<RouteRoadDressing>(roadObject);
        else Undo.RecordObject(dressing, "Road dressing");
        dressing.roadMesh = road;
        dressing.sidewalkMaterial = LoadMat("Sidewalk");
        dressing.curbMaterial = LoadMat("Curb");
        dressing.centreLineMaterial = LoadMat("Line_White");
        dressing.edgeLineMaterial = LoadMat("Line_White");
        var mr = roadObject.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            Undo.RecordObject(mr, "Road material");
            mr.sharedMaterial = LoadMat("Asphalt");
        }
        EditorUtility.SetDirty(dressing);
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        return dressing;
    }

    [MenuItem("Tools/Environment Kit/Create showcase scene")]
    public static void CreateShowcase()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        EnsureKit();
        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        ApplyLook();

        var cam = new GameObject("Main Camera");
        cam.tag = "MainCamera";
        var c = cam.AddComponent<Camera>();
        c.farClipPlane = 800f;
        cam.AddComponent<AudioListener>();
        cam.transform.position = new Vector3(-10f, 45f, -40f);
        cam.transform.LookAt(new Vector3(70f, 0f, 40f));

        var parent = new GameObject("Showcase").transform;
        Put("GroundPlane", 0f, 0f, 0f, parent);

        // every prefab in rows, in catalogue order: people, vehicles, buildings, trees, bushes, street furniture
        BuildCatalog();
        float x = 0f, z = 0f, rowDepth = 0f;
        foreach (Item it in Catalog)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PreDir + "/" + it.name + ".prefab");
            if (prefab == null) continue;
            GameObject g = Put(it.name, 0f, 0f, 0f, parent);
            var col = g.GetComponent<Collider>();
            Vector3 size = col != null ? col.bounds.size : Vector3.one;
            float w = Mathf.Max(size.x, 1f), d = Mathf.Max(size.z, 1f);
            if (x + w > 160f) { x = 0f; z += rowDepth + 8f; rowDepth = 0f; }
            g.transform.position = new Vector3(x + w * 0.5f, 0f, z);
            x += w + 2f;
            rowDepth = Mathf.Max(rowDepth, d);
        }

        EnsureDir("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, ShowcasePath);
        UnityEngine.Debug.Log("ENVIRONMENT KIT: showcase scene saved as " + ShowcasePath + " (every model faces +Z, the blue axis arrow of the move tool)");
    }

    static void EnsureKit()
    {
        if (AssetDatabase.LoadAssetAtPath<Material>(MatDir + "/Sky_Day.mat") == null || AssetDatabase.LoadAssetAtPath<GameObject>(PreDir + "/GroundPlane.prefab") == null)
            BuildKit();
    }

    // ---- files -------------------------------------------------------------------------------------------------------------------------

    public static void EnsureDir(string assetDir)
    {
        string abs = Path.Combine(Directory.GetCurrentDirectory(), assetDir);
        if (!Directory.Exists(abs)) Directory.CreateDirectory(abs);
    }

    static Material LoadMat(string name)
    {
        var m = AssetDatabase.LoadAssetAtPath<Material>(MatDir + "/" + name + ".mat");
        if (m == null) UnityEngine.Debug.LogError("ENVIRONMENT KIT: material '" + name + "' is missing, run Tools > Environment Kit > Build kit");
        return m;
    }

    // ---- textures and materials --------------------------------------------------------------------------------------------------------

    static Color G(float v) { return new Color(v, v, v, 1f); }

    // tileable value noise (the lattice wraps), so the textures have no seams
    static float H(int x, int y, int seed)
    {
        unchecked
        {
            int h = x * 374761393 ^ y * 668265263 ^ seed * 1442695041;
            h = (h ^ (h >> 13)) * 1274126177;
            h ^= h >> 16;
            return (h & 0xffff) / 65535f;
        }
    }

    static float VNoise(float x, float y, int period, int seed)
    {
        int x0 = Mathf.FloorToInt(x), y0 = Mathf.FloorToInt(y);
        float fx = x - x0, fy = y - y0;
        fx = fx * fx * (3f - 2f * fx);
        fy = fy * fy * (3f - 2f * fy);
        int xa = ((x0 % period) + period) % period, xb = (xa + 1) % period;
        int ya = ((y0 % period) + period) % period, yb = (ya + 1) % period;
        float top = Mathf.Lerp(H(xa, ya, seed), H(xb, ya, seed), fx);
        float bot = Mathf.Lerp(H(xa, yb, seed), H(xb, yb, seed), fx);
        return Mathf.Lerp(top, bot, fy);
    }

    static Color NoisePx(Color c, int x, int y, int seed, float coarse, float medium, float fine)
    {
        float u = x / (float)TexSize, v = y / (float)TexSize;
        float n = (VNoise(u * 8f, v * 8f, 8, seed) - 0.5f) * coarse
                + (VNoise(u * 32f, v * 32f, 32, seed + 1) - 0.5f) * medium
                + (H(x, y, seed + 2) - 0.5f) * fine;
        float f = Mathf.Max(0f, 1f + n);
        return new Color(c.r * f, c.g * f, c.b * f, 1f);
    }

    static Texture2D MakeTexture(string name, Func<int, int, Color> px)
    {
        EnsureDir(TexDir);
        string path = TexDir + "/" + name + ".png";
        var t = new Texture2D(TexSize, TexSize, TextureFormat.RGBA32, false, false);
        var cols = new Color32[TexSize * TexSize];
        for (int y = 0; y < TexSize; y++)
            for (int x = 0; x < TexSize; x++)
                cols[y * TexSize + x] = px(x, y);
        t.SetPixels32(cols);
        t.Apply();
        File.WriteAllBytes(Path.Combine(Directory.GetCurrentDirectory(), path), t.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(t);
        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
        var imp = (TextureImporter)AssetImporter.GetAtPath(path);
        imp.wrapMode = TextureWrapMode.Repeat;
        imp.filterMode = FilterMode.Bilinear;
        imp.mipmapEnabled = true;
        imp.anisoLevel = 8;
        imp.SaveAndReimport();
        return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
    }

    static Material Mat(string name, Color c, Texture2D tex = null, float tiling = 1f, float glossiness = 0.05f)
    {
        EnsureDir(MatDir);
        string path = MatDir + "/" + name + ".mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        bool isNew = m == null;
        if (isNew) m = new Material(Shader.Find("Standard"));
        m.color = tex != null ? Color.white : c;
        m.mainTexture = tex;
        m.mainTextureScale = new Vector2(tiling, tiling);
        if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", glossiness);
        if (m.HasProperty("_Metallic")) m.SetFloat("_Metallic", 0f);
        if (isNew) AssetDatabase.CreateAsset(m, path); else EditorUtility.SetDirty(m);
        return m;
    }

    // Colours follow the Pandazole road tiles (dark asphalt, light concrete, white paint) so the driven road matches the pack's look.
    // The textures add the detail the pack's flat colours do not have: speckle and patches in the asphalt, slab joints in the sidewalk.
    static void BuildMaterials()
    {
        Color asphalt = new Color(0.17f, 0.18f, 0.20f), walk = new Color(0.70f, 0.70f, 0.70f), curb = new Color(0.76f, 0.76f, 0.76f);
        Mat("Asphalt", asphalt, MakeTexture("T_Asphalt", (x, y) => NoisePx(asphalt, x, y, 11, 0.35f, 0.25f, 0.30f)));            // 4 m tile (RouteRoadMesh.uvTileM)
        Mat("Sidewalk", walk, MakeTexture("T_Sidewalk", (x, y) =>
            (x < 3 || y < 3) ? G(0.52f) : NoisePx(walk, x, y, 23, 0.12f, 0.10f, 0.14f)));                                          // 2 m slabs (RouteRoadDressing.uvTileM)
        Mat("Curb", curb, MakeTexture("T_Curb", (x, y) => NoisePx(curb, x, y, 37, 0.10f, 0.10f, 0.14f)));
        Mat("Line_White", new Color(0.93f, 0.93f, 0.93f));
        // fallback ground (the scenes use their own ground material) and the green disc under trees
        Mat("Ground", G(0.42f), null, 1f, 0f);
        Mat("Grass_Patch", new Color(0.36f, 0.52f, 0.26f), null, 1f, 0f);
        Mat("Dolly_Base", G(0.22f));
        Mat("Dolly_Wheel", G(0.08f));
        Mat("Dolly_Pole", new Color(1.00f, 0.45f, 0.05f));

        Shader sky = Shader.Find("Skybox/Procedural");
        string skyPath = MatDir + "/Sky_Day.mat";
        var sm = AssetDatabase.LoadAssetAtPath<Material>(skyPath);
        if (sm == null) { sm = new Material(sky); AssetDatabase.CreateAsset(sm, skyPath); }
        if (sm.HasProperty("_Exposure")) sm.SetFloat("_Exposure", 1.2f);
        EditorUtility.SetDirty(sm);
    }

    // ---- wrapper prefabs ---------------------------------------------------------------------------------------------------------------

    static Bounds WorldBounds(GameObject g)
    {
        Renderer[] rs = g.GetComponentsInChildren<Renderer>();
        Bounds b = new Bounds();
        bool any = false;
        foreach (Renderer r in rs)
        {
            if (!any) { b = r.bounds; any = true; } else b.Encapsulate(r.bounds);
        }
        return b;
    }

    static bool Wrap(Item it)
    {
        string sourcePath = ResolveSource(it.source);
        var src = sourcePath == null ? null : AssetDatabase.LoadAssetAtPath<GameObject>(sourcePath);
        if (src == null)
        {
            UnityEngine.Debug.LogWarning("ENVIRONMENT KIT: source not found for " + it.name + ": " + it.source);
            return false;
        }
        float yaw = it.yaw, scale = it.scale;
        bool kept = false;
        if (!resetOrientation)
        {
            var old = AssetDatabase.LoadAssetAtPath<GameObject>(PreDir + "/" + it.name + ".prefab");
            Transform ov = old != null ? old.transform.Find("Visual") : null;
            if (ov != null) { yaw = ov.localEulerAngles.y; scale = ov.localScale.x; kept = true; }
        }
        var root = new GameObject(it.name);
        var visual = UnityEngine.Object.Instantiate(src);        // a plain copy, not a prefab instance: its colliders can be removed
        visual.name = "Visual";
        visual.transform.SetParent(root.transform, false);
        visual.transform.localPosition = Vector3.zero;
        visual.transform.localRotation = Quaternion.Euler(0f, yaw, 0f);
        visual.transform.localScale = Vector3.one * scale;
        foreach (Collider c in visual.GetComponentsInChildren<Collider>()) UnityEngine.Object.DestroyImmediate(c);
        if (it.animal) PrepareAnimal(visual, it.name, AnimalFolder(it.source));
        if (it.fitLength > 0f && !kept)
        {
            visual.transform.localScale = Vector3.one;
            Bounds fb = WorldBounds(root);
            float longest = Mathf.Max(fb.size.x, fb.size.z);
            if (longest > 0.001f) { scale = it.fitLength / longest; visual.transform.localScale = Vector3.one * scale; }
        }
        if (it.aPose) MakeAPose(visual, 40f);
        if (it.yellow) TintBody(visual);

        Bounds b = WorldBounds(root);
        float k = scale;
        float lift = it.dolly ? 0.12f * k : 0f;                  // the model stands on the base
        Vector3 shift = new Vector3(it.capsule ? 0f : -b.center.x, -b.min.y + lift, it.capsule ? 0f : -b.center.z);
        visual.transform.localPosition = shift;
        float modelHeight = b.size.y;
        if (it.dolly) AddDolly(root, k, lift, modelHeight);
        if (it.patch > 0f) AddPatch(root, it.patch);
        b = WorldBounds(root);

        if (it.capsule)
        {
            float h = b.size.y * it.heightFrac;
            var cc = root.AddComponent<CapsuleCollider>();
            cc.direction = 1;
            cc.radius = it.radius;
            cc.height = Mathf.Max(h, 2f * it.radius);
            cc.center = new Vector3(0f, cc.height * 0.5f, 0f);
        }
        else
        {
            var bc = root.AddComponent<BoxCollider>();
            bc.center = b.center;
            bc.size = b.size;
        }
        root.AddComponent<Hazard>().label = it.label;
        SetLayer(root, LayerHazard);
        Save(root);
        return true;
    }

    // arms down: the humanoid models come in a T pose. Each upper arm is swung down to 'deg' degrees below horizontal. The Animator is
    // removed afterwards (it has no controller anyway), so the pose stays as saved.
    static void MakeAPose(GameObject visual, float deg)
    {
        var anim = visual.GetComponentInChildren<Animator>();
        if (anim == null || anim.avatar == null || !anim.avatar.isHuman)
        {
            UnityEngine.Debug.LogWarning("ENVIRONMENT KIT: '" + visual.name + "' has no humanoid Animator, no A pose made");
            return;
        }
        PoseArm(anim, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, deg);
        PoseArm(anim, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, deg);
        UnityEngine.Object.DestroyImmediate(anim);
    }

    static void PoseArm(Animator a, HumanBodyBones upper, HumanBodyBones lower, float deg)
    {
        Transform u = a.GetBoneTransform(upper), l = a.GetBoneTransform(lower);
        if (u == null || l == null) return;
        Vector3 dir = (l.position - u.position).normalized;
        Vector3 flat = new Vector3(dir.x, 0f, dir.z);
        if (flat.sqrMagnitude < 1e-6f) return;
        flat.Normalize();
        float rad = deg * Mathf.Deg2Rad;
        Vector3 target = flat * Mathf.Cos(rad) + Vector3.down * Mathf.Sin(rad);
        u.rotation = Quaternion.FromToRotation(dir, target) * u.rotation;
    }

    // the body material of the dummy (white in the pack) gets a crash-test yellow tint; the texture's dark details stay
    static void TintBody(GameObject visual)
    {
        foreach (Renderer r in visual.GetComponentsInChildren<Renderer>())
        {
            Material[] mats = r.sharedMaterials;
            bool changed = false;
            for (int i = 0; i < mats.Length; i++)
            {
                if (mats[i] == null || mats[i].name != "Body") continue;
                string path = MatDir + "/Dummy_Yellow.mat";
                var m = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (m == null) { m = new Material(mats[i]); AssetDatabase.CreateAsset(m, path); }
                else if (m.shader != mats[i].shader || m.mainTexture != mats[i].mainTexture) m.CopyPropertiesFromMaterial(mats[i]);
                m.color = new Color(1f, 0.82f, 0f);
                if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 0.05f);
                EditorUtility.SetDirty(m);
                mats[i] = m;
                changed = true;
            }
            if (changed) r.sharedMaterials = mats;
        }
    }

    // a flat green disc on the ground around the foot of a tree
    static void AddPatch(GameObject root, float diameter)
    {
        var d = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        d.name = "GrassPatch";
        d.transform.SetParent(root.transform, false);
        d.transform.localPosition = new Vector3(0f, 0.01f, 0f);
        d.transform.localScale = new Vector3(diameter, 0.01f, diameter);                  // height 0.02
        d.GetComponent<MeshRenderer>().sharedMaterial = LoadMat("Grass_Patch");
        UnityEngine.Object.DestroyImmediate(d.GetComponent<Collider>());
    }

    // a flat round base on four small wheels with a pole that goes up into the middle of the model
    static void AddDolly(GameObject root, float k, float lift, float modelHeight)
    {
        Material baseMat = LoadMat("Dolly_Base"), wheelMat = LoadMat("Dolly_Wheel"), poleMat = LoadMat("Dolly_Pole");
        var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        disc.name = "DollyBase";
        disc.transform.SetParent(root.transform, false);
        disc.transform.localPosition = new Vector3(0f, lift - 0.02f * k, 0f);
        disc.transform.localScale = new Vector3(0.55f * k, 0.02f * k, 0.55f * k);          // diameter 0.55 k, height 0.04 k
        disc.GetComponent<MeshRenderer>().sharedMaterial = baseMat;
        UnityEngine.Object.DestroyImmediate(disc.GetComponent<Collider>());
        foreach (float sx in new[] { -1f, 1f })
            foreach (float sz in new[] { -1f, 1f })
            {
                var w = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                w.name = "DollyWheel";
                w.transform.SetParent(root.transform, false);
                w.transform.localPosition = new Vector3(sx * 0.17f * k, 0.05f * k, sz * 0.17f * k);
                w.transform.localEulerAngles = new Vector3(0f, 0f, 90f);
                w.transform.localScale = new Vector3(0.10f * k, 0.025f * k, 0.10f * k);   // diameter 0.1 k, width 0.05 k
                w.GetComponent<MeshRenderer>().sharedMaterial = wheelMat;
                UnityEngine.Object.DestroyImmediate(w.GetComponent<Collider>());
            }
        float poleTop = lift + 0.55f * modelHeight;
        float poleH = poleTop - lift;
        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "DollyPole";
        pole.transform.SetParent(root.transform, false);
        pole.transform.localPosition = new Vector3(0f, lift + poleH * 0.5f, 0f);
        pole.transform.localScale = new Vector3(0.05f * k, poleH * 0.5f, 0.05f * k);
        pole.GetComponent<MeshRenderer>().sharedMaterial = poleMat;
        UnityEngine.Object.DestroyImmediate(pole.GetComponent<Collider>());
    }

    static void SetLayer(GameObject g, int layer)
    {
        g.layer = layer;
        foreach (Transform t in g.transform) SetLayer(t.gameObject, layer);
    }

    // "find:<folder>:<name>" -> the shortest prefab (else model) file name that contains <name> below Assets/Asset Packs/<folder>; other strings are paths
    static string ResolveSource(string source)
    {
        if (!source.StartsWith("find:", StringComparison.Ordinal)) return source;
        string[] p = source.Split(':');
        string best = null;
        foreach (string type in new[] { "t:Prefab", "t:Model" })
        {
            foreach (string guid in AssetDatabase.FindAssets(p[2] + " " + type, new[] { Packs + p[1] }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                string n = Path.GetFileNameWithoutExtension(path);
                if (n.IndexOf(p[2], StringComparison.OrdinalIgnoreCase) < 0) continue;
                if (best == null) { best = path; continue; }
                string bn = Path.GetFileNameWithoutExtension(best);
                if (n.Length < bn.Length || (n.Length == bn.Length && string.CompareOrdinal(n, bn) < 0)) best = path;
            }
            if (best != null) break;
        }
        UnityEngine.Debug.Log("ENVIRONMENT KIT: '" + source + "' resolved to " + (best ?? "NOTHING"));
        return best;
    }

    // animals: the pack's scripts, character controller and rigid body are removed (the Animator stays, so the model keeps its animation controller);
    // the pack's materials are for another render pipeline, so each one is re-made as a Standard material with the same texture and colour
    static void PrepareAnimal(GameObject visual, string name, string folder)
    {
        foreach (MonoBehaviour mb in visual.GetComponentsInChildren<MonoBehaviour>(true)) UnityEngine.Object.DestroyImmediate(mb);
        foreach (CharacterController cc in visual.GetComponentsInChildren<CharacterController>(true)) UnityEngine.Object.DestroyImmediate(cc);
        foreach (Rigidbody rb in visual.GetComponentsInChildren<Rigidbody>(true)) UnityEngine.Object.DestroyImmediate(rb);
        EnsureDir(MatDir);
        SetupAnimation(visual, name, folder);
        int k = 0;
        foreach (Renderer r in visual.GetComponentsInChildren<Renderer>(true))
        {
            Material[] mats = r.sharedMaterials;
            for (int i = 0; i < mats.Length; i++)
            {
                if (mats[i] == null) continue;
                Texture tex = null;
                Color col = Color.white;
                var so = new SerializedObject(mats[i]);
                SerializedProperty texes = so.FindProperty("m_SavedProperties.m_TexEnvs");
                for (int t = 0; t < texes.arraySize; t++)
                {
                    string pn = texes.GetArrayElementAtIndex(t).FindPropertyRelative("first").stringValue;
                    var tp = texes.GetArrayElementAtIndex(t).FindPropertyRelative("second.m_Texture");
                    if ((pn == "_BaseMap" || pn == "_MainTex") && tp.objectReferenceValue != null) { tex = (Texture)tp.objectReferenceValue; break; }
                }
                SerializedProperty cols = so.FindProperty("m_SavedProperties.m_Colors");
                for (int c = 0; c < cols.arraySize; c++)
                {
                    string pn = cols.GetArrayElementAtIndex(c).FindPropertyRelative("first").stringValue;
                    if (pn == "_BaseColor" || pn == "_Color") { col = cols.GetArrayElementAtIndex(c).FindPropertyRelative("second").colorValue; break; }
                }
                string path = MatDir + "/" + name + "_" + (k++) + ".mat";
                var m = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (m == null) { m = new Material(Shader.Find("Standard")); AssetDatabase.CreateAsset(m, path); }
                m.mainTexture = tex;
                m.color = col;
                if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", 0.05f);
                if (tex == null) tex = FindPackTexture(folder, mats[i].name, name);
                if (tex == null) UnityEngine.Debug.LogWarning("ENVIRONMENT KIT: no texture found for material '" + mats[i].name + "' of " + name + ", it is plain colour");
                EditorUtility.SetDirty(m);
                mats[i] = m;
            }
            r.sharedMaterials = mats;
        }
    }

    static string AnimalFolder(string source)
    {
        string[] p = source.Split(':');
        return Packs + p[1];
    }

    // the texture of a pack material that has none assigned: a texture file below the pack folder whose name contains the material's or the model's
    // name (first the material), else the only texture there is
    static Texture FindPackTexture(string folder, string materialName, string modelName)
    {
        var all = new List<string>();
        foreach (string guid in AssetDatabase.FindAssets("t:Texture2D", new[] { folder })) all.Add(AssetDatabase.GUIDToAssetPath(guid));
        string pick = null;
        foreach (string key in new[] { materialName, "kitty", "animal", "cat" })
        {
            foreach (string path in all)
                if (Path.GetFileNameWithoutExtension(path).IndexOf(key, StringComparison.OrdinalIgnoreCase) >= 0) { pick = path; break; }
            if (pick != null) break;
        }
        if (pick == null && all.Count == 1) pick = all[0];
        UnityEngine.Debug.Log("ENVIRONMENT KIT: texture for '" + materialName + "': " + (pick ?? "none found among " + all.Count + " textures in " + folder));
        return pick == null ? null : AssetDatabase.LoadAssetAtPath<Texture>(pick);
    }

    // an Animator controller with one looping state playing a run clip (else a walk clip) found below the pack folder; the model plays it in place
    static void SetupAnimation(GameObject visual, string name, string folder)
    {
        var anim = visual.GetComponentInChildren<Animator>();
        if (anim == null) { UnityEngine.Debug.LogWarning("ENVIRONMENT KIT: " + name + " has no Animator, no animation"); return; }
        AnimationClip best = null;
        foreach (string key in new[] { "run", "walk", "" })
        {
            foreach (string guid in AssetDatabase.FindAssets("t:AnimationClip", new[] { folder }))
                foreach (UnityEngine.Object o in AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GUIDToAssetPath(guid)))
                {
                    var c = o as AnimationClip;
                    if (c == null || c.name.StartsWith("__preview__", StringComparison.Ordinal)) continue;
                    if (key.Length > 0 && c.name.IndexOf(key, StringComparison.OrdinalIgnoreCase) < 0) continue;
                    best = c; break;
                }
            if (best != null) break;
        }
        if (best == null) { UnityEngine.Debug.LogWarning("ENVIRONMENT KIT: no animation clip found below " + folder); return; }
        string path = MatDir.Replace("/Materials", "") + "/" + name + "_Anim.controller";
        var ctrl = UnityEditor.Animations.AnimatorController.CreateAnimatorControllerAtPath(path);
        var st = ctrl.layers[0].stateMachine.AddState("Move");
        st.motion = best;
        ctrl.layers[0].stateMachine.defaultState = st;
        anim.runtimeAnimatorController = ctrl;
        anim.applyRootMotion = false;
        UnityEngine.Debug.Log("ENVIRONMENT KIT: " + name + " plays the clip '" + best.name + "'");
    }

    static void Save(GameObject root)
    {
        EnsureDir(PreDir);
        PrefabUtility.SaveAsPrefabAsset(root, PreDir + "/" + root.name + ".prefab");
        UnityEngine.Object.DestroyImmediate(root);
    }

    // the ground: 400 x 400 m, top surface at y = -0.02 (2 cm under the roads), Ground layer, no Hazard. Built from the first call of Build kit.
    static void BuildGround()
    {
        var r = new GameObject("GroundPlane");
        var v = GameObject.CreatePrimitive(PrimitiveType.Cube);
        v.name = "Slab";
        v.transform.SetParent(r.transform, false);
        v.transform.localPosition = new Vector3(0f, -0.52f, 0f);
        v.transform.localScale = new Vector3(400f, 1f, 400f);
        UnityEngine.Object.DestroyImmediate(v.GetComponent<Collider>());
        var own = AssetDatabase.LoadAssetAtPath<Material>("Assets/Materials/Ground.mat");      // the ground material the scenes already use
        v.GetComponent<MeshRenderer>().sharedMaterial = own != null ? own : LoadMat("Ground");
        var bc = r.AddComponent<BoxCollider>();
        bc.center = new Vector3(0f, -0.52f, 0f);
        bc.size = new Vector3(400f, 1f, 400f);
        SetLayer(r, LayerGround);
        Save(r);
    }

    static GameObject Put(string prefab, float x, float y, float z, Transform parent)
    {
        var p = AssetDatabase.LoadAssetAtPath<GameObject>(PreDir + "/" + prefab + ".prefab");
        var g = (GameObject)PrefabUtility.InstantiatePrefab(p);
        g.transform.SetParent(parent, false);
        g.transform.position = new Vector3(x, y, z);
        return g;
    }

}