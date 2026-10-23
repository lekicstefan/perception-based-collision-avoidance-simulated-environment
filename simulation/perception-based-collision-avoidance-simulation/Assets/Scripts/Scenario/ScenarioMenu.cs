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
        UiTheme.Begin();
        DrawUi();
        UiTheme.End();
    }

    void DrawUi()
    {
        UiTheme.Ensure();
        UiTheme.Fill(new Rect(0f, 0f, UiTheme.W, UiTheme.H), new Color32(10, 14, 19, 255));      // backdrop

        float w = Mathf.Min(UiTheme.W - 48f, 900f);
        float x = (UiTheme.W - w) * 0.5f;

        // header: title and the count
        Rect[] head = UiTheme.Cols(new Rect(x, 28f, w, 32f), 0f, 260f);
        GUI.Label(head[0], "Scenarios", UiTheme.Name);
        GUI.Label(head[1], entries.Count + " files", UiTheme.Right);
        UiTheme.Separator(new Rect(x, 64f, w, 1f));

        // the list: every entry is a row with Start, the name and the scene, and a line of description below
        const float entryH = 66f;
        var view = new Rect(x, 72f, w, UiTheme.H - 72f - 24f);
        var content = new Rect(0f, 0f, w - 16f, entries.Count * entryH);
        scroll = GUI.BeginScrollView(view, scroll, content);
        for (int i = 0; i < entries.Count; i++)
        {
            Entry e = entries[i];
            float y = i * entryH;
            Rect[] c = UiTheme.Cols(new Rect(0f, y + 8f, content.width, UiTheme.RowH), 74f, 0f, 240f);
            GUI.enabled = e.canLoad;
            if (UiTheme.Press(c[0], "Start"))
            {
                ScenarioContext.Pick = e.name;
                UnityEngine.Debug.Log("MENU | starting scenario " + e.name + " in scene " + e.scene);
                SceneManager.LoadScene(e.scene);
            }
            GUI.enabled = true;
            GUI.Label(c[1], e.name, UiTheme.Name);
            GUI.Label(c[2], e.scene ?? "", UiTheme.Right);

            var line = new Rect(82f, y + 8f + UiTheme.RowH + 4f, content.width - 82f, 28f);
            if (!string.IsNullOrEmpty(e.error)) GUI.Label(line, e.error, new GUIStyle(UiTheme.Wrap) { normal = { textColor = UiTheme.Red } });
            else GUI.Label(line, e.description ?? "", UiTheme.Wrap);
            UiTheme.Separator(new Rect(0f, y + entryH - 1f, content.width, 1f));
        }
        GUI.EndScrollView();
    }
}