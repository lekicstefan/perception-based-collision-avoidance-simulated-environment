using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using static System.Net.Mime.MediaTypeNames;

// The scenario menu: put this on an empty object in a scene called "Menu" (ScenarioContext.MenuScene) and list that scene first in the Build Settings.
// It lists every file in configs/scenarios with its scene and description. Start loads the scene of the scenario and makes the scenario the one
// that the scene runs (ScenarioContext.Pick), so a restart (R) repeats it and "Menu (M)" in a scenario scene comes back here.
// The processor has to be running (with --loop, so that it waits for the next run) before a scenario is started: the scene waits for it.
// A scenario whose scene is not in the Build Settings is greyed out. Debug only, never sent to the processor.
public class ScenarioMenu : MonoBehaviour
{
    class Entry
    {
        public string name, scene, description, error;
        public bool canLoad;
    }

    readonly List<Entry> entries = new List<Entry>();
    Vector2 scroll;
    GUIStyle title, small, errorStyle;

    void Awake()
    {
        Time.timeScale = 1f;       // the scene that came before may have been frozen
        Refresh();
    }

    void Refresh()
    {
        entries.Clear();
        foreach (string name in ScenarioLoader.Available())
        {
            var e = new Entry { name = name };
            try
            {
                string path;
                ScenarioConfig c = ScenarioLoader.Load(name, new List<string>(), out path);
                e.scene = c.scene;
                e.description = c.description;
                e.canLoad = UnityEngine.Application.CanStreamedLevelBeLoaded(c.scene);
                if (!e.canLoad) e.error = "scene '" + c.scene + "' is not in the Build Settings";
            }
            catch (Exception ex)
            {
                e.error = ex.Message;
            }
            entries.Add(e);
        }
        entries.Sort((a, b) => CompareNatural(a.name, b.name));
    }

    // s2_... before s10_...
    static int CompareNatural(string a, string b)
    {
        int na = LeadingNumber(a), nb = LeadingNumber(b);
        if (na != nb) return na.CompareTo(nb);
        return string.CompareOrdinal(a, b);
    }

    static int LeadingNumber(string s)
    {
        int i = 0;
        while (i < s.Length && !char.IsDigit(s[i])) i++;
        int v = 0, digits = 0;
        while (i < s.Length && char.IsDigit(s[i]) && digits < 6) { v = v * 10 + (s[i] - '0'); i++; digits++; }
        return digits == 0 ? int.MaxValue : v;
    }

    void Start()
    {
        if (!string.IsNullOrEmpty(ScenarioContext.Pick))
            UnityEngine.Debug.Log("MENU | last scenario: " + ScenarioContext.Pick);
    }

    void OnGUI()
    {
        if (title == null)
        {
            title = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold };
            small = new GUIStyle(GUI.skin.label) { fontSize = 11, wordWrap = true };
            errorStyle = new GUIStyle(small);
            errorStyle.normal.textColor = new Color(1f, 0.4f, 0.3f);
        }

        float w = Mathf.Min(Screen.width - 40f, 900f);
        GUILayout.BeginArea(new Rect((Screen.width - w) * 0.5f, 20f, w, Screen.height - 40f));
        GUILayout.Label("Scenarios", title);
        GUILayout.Label("Start the processor first (python -m server.run_processor --pin 2,4 --run-dir NAME --loop), then pick a scenario. " +
                        "In a scenario: R restart, P pause, X end the run, M back to this menu.", small);
        string last = string.IsNullOrEmpty(ScenarioContext.Pick) ? "none yet" : ScenarioContext.Pick;
        GUILayout.Label("Last scenario: " + last + "   (" + entries.Count + " scenario files)", small);
        if (GUILayout.Button("Reload the list", GUILayout.Width(130f))) Refresh();
        GUILayout.Space(6f);

        scroll = GUILayout.BeginScrollView(scroll);
        foreach (Entry e in entries)
        {
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.BeginHorizontal();
            GUI.enabled = e.canLoad;
            if (GUILayout.Button("Start", GUILayout.Width(70f), GUILayout.Height(24f)))
            {
                ScenarioContext.Pick = e.name;
                UnityEngine.Debug.Log("MENU | starting scenario " + e.name + " in scene " + e.scene);
                SceneManager.LoadScene(e.scene);
            }
            GUI.enabled = true;
            GUILayout.Label(e.name + (e.scene != null ? "   [" + e.scene + "]" : ""));
            GUILayout.EndHorizontal();
            if (!string.IsNullOrEmpty(e.error)) GUILayout.Label(e.error, errorStyle);
            if (!string.IsNullOrEmpty(e.description))
                GUILayout.Label(e.description.Length > 260 ? e.description.Substring(0, 257) + "..." : e.description, small);
            GUILayout.EndVertical();
        }
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }
}