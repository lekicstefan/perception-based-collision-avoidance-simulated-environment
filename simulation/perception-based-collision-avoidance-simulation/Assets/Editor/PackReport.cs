using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

// Lists what the imported asset packs contain (everything under Assets except our own folders) into environment_pack_report.txt in the
// project root: prefabs (size, colliders, shaders), model files of the packs that have no prefabs, textures and materials.
// Read-only, changes nothing.
public static class PackReport
{
    static readonly string[] Skip = { "Assets/Scripts/", "Assets/Editor/", "Assets/Environment/", "Assets/Scenes/" };
    const int MaxLines = 400;

    static bool Skipped(string path)
    {
        foreach (string s in Skip) if (path.StartsWith(s, StringComparison.Ordinal)) return true;
        return false;
    }

    [MenuItem("Tools/Environment Kit/Report asset packs")]
    public static void Run()
    {
        var sb = new StringBuilder();
        sb.AppendLine("unity " + UnityEngine.Application.unityVersion);
        sb.AppendLine("render pipeline: " + (GraphicsSettings.currentRenderPipeline != null ? GraphicsSettings.currentRenderPipeline.name : "Built-in"));
        sb.AppendLine();

        // 1. prefabs
        var byDir = new SortedDictionary<string, List<string>>();
        int count = 0;
        foreach (string g in AssetDatabase.FindAssets("t:Prefab", new[] { "Assets" }))
        {
            string path = AssetDatabase.GUIDToAssetPath(g);
            if (Skipped(path)) continue;
            string dir = Path.GetDirectoryName(path).Replace('\\', '/');
            if (!byDir.ContainsKey(dir)) byDir[dir] = new List<string>();
            byDir[dir].Add(DescribePrefab(path));
            count++;
        }
        sb.AppendLine("== prefabs: " + count + " ==");
        foreach (var kv in byDir)
        {
            sb.AppendLine();
            sb.AppendLine("[" + kv.Key + "] " + kv.Value.Count + " prefabs");
            foreach (string line in kv.Value) sb.AppendLine("  " + line);
        }

        // 2. model files of the packs that were not covered by prefabs (Banana Man, Kenney)
        sb.AppendLine();
        sb.AppendLine("== model files in Banana / Kenney folders ==");
        int lines = 0;
        foreach (string g in AssetDatabase.FindAssets("t:Model", new[] { "Assets" }))
        {
            string path = AssetDatabase.GUIDToAssetPath(g);
            if (Skipped(path) || !(path.Contains("Banana") || path.Contains("Kenney"))) continue;
            if (++lines > MaxLines) { sb.AppendLine("  ... more"); break; }
            sb.AppendLine("  " + DescribeModel(path));
        }

        // 3. textures
        sb.AppendLine();
        sb.AppendLine("== textures (path | width x height) ==");
        lines = 0;
        foreach (string g in AssetDatabase.FindAssets("t:Texture2D", new[] { "Assets" }))
        {
            string path = AssetDatabase.GUIDToAssetPath(g);
            if (Skipped(path)) continue;
            if (++lines > MaxLines) { sb.AppendLine("  ... more"); break; }
            var t = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            sb.AppendLine("  " + path + " | " + (t != null ? t.width + " x " + t.height : "?"));
        }

        // 4. materials
        sb.AppendLine();
        sb.AppendLine("== materials (path | shader | main texture) ==");
        lines = 0;
        foreach (string g in AssetDatabase.FindAssets("t:Material", new[] { "Assets" }))
        {
            string path = AssetDatabase.GUIDToAssetPath(g);
            if (Skipped(path)) continue;
            if (++lines > MaxLines) { sb.AppendLine("  ... more"); break; }
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null) continue;
            string tex = m.mainTexture != null ? m.mainTexture.name : "none";
            sb.AppendLine("  " + path + " | " + (m.shader != null ? m.shader.name : "?") + (m.shader != null && !m.shader.isSupported ? " (NOT SUPPORTED)" : "") + " | " + tex);
        }

        string file = Path.Combine(Directory.GetCurrentDirectory(), "environment_pack_report.txt");
        File.WriteAllText(file, sb.ToString());
        UnityEngine.Debug.Log("PACK REPORT: written to " + file);
    }

    static string Measure(GameObject root, out int meshes)
    {
        Renderer[] rs = root.GetComponentsInChildren<Renderer>();
        meshes = rs.Length;
        Bounds b = new Bounds();
        bool any = false;
        foreach (Renderer r in rs)
        {
            if (!any) { b = r.bounds; any = true; } else b.Encapsulate(r.bounds);
        }
        return any ? b.size.x.ToString("0.0") + " x " + b.size.y.ToString("0.0") + " x " + b.size.z.ToString("0.0") : "no renderer";
    }

    static string ShaderList(GameObject root)
    {
        var shaders = new SortedSet<string>();
        foreach (Renderer r in root.GetComponentsInChildren<Renderer>())
            foreach (Material m in r.sharedMaterials)
            {
                if (m != null && m.shader != null) shaders.Add(m.shader.name + (m.shader.isSupported ? "" : " (NOT SUPPORTED)"));
                else shaders.Add("(missing material)");
            }
        return string.Join("; ", shaders);
    }

    static string DescribePrefab(string path)
    {
        GameObject root = null;
        try
        {
            root = PrefabUtility.LoadPrefabContents(path);
            int meshes;
            string size = Measure(root, out meshes);
            var colliders = new Dictionary<string, int>();
            foreach (Collider c in root.GetComponentsInChildren<Collider>())
            {
                string k = c.GetType().Name.Replace("Collider", "");
                colliders[k] = colliders.ContainsKey(k) ? colliders[k] + 1 : 1;
            }
            var parts = new List<string>();
            foreach (var kv in colliders) parts.Add(kv.Value + " " + kv.Key);
            string col = parts.Count == 0 ? "no colliders" : string.Join(",", parts);
            string extra = (root.GetComponentInChildren<Animator>() != null ? " animator" : "") + " meshes:" + meshes;
            return Path.GetFileNameWithoutExtension(path) + " | size x,y,z " + size + " | " + col + " | " + ShaderList(root) + extra;
        }
        catch (Exception e)
        {
            return Path.GetFileNameWithoutExtension(path) + " | error: " + e.Message;
        }
        finally
        {
            if (root != null) PrefabUtility.UnloadPrefabContents(root);
        }
    }

    static string DescribeModel(string path)
    {
        GameObject inst = null;
        try
        {
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (model == null) return path + " | not a GameObject";
            inst = UnityEngine.Object.Instantiate(model);
            int meshes;
            string size = Measure(inst, out meshes);
            var imp = AssetImporter.GetAtPath(path) as ModelImporter;
            string rig = imp != null ? imp.animationType.ToString() : "?";
            return path + " | size x,y,z " + size + " | rig " + rig + " | " + ShaderList(inst) + " meshes:" + meshes;
        }
        catch (Exception e)
        {
            return path + " | error: " + e.Message;
        }
        finally
        {
            if (inst != null) UnityEngine.Object.DestroyImmediate(inst);
        }
    }
}