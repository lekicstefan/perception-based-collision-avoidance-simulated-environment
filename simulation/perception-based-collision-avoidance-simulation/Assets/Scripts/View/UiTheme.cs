using UnityEngine;

// The look of the on-screen panels (run controls, view switches, scenario menu, dashboard). Flat and plain: one dark panel colour, one thin line
// colour, one accent, regular weight text, no rounded corners, no boxes inside boxes. Debug UI only (IMGUI), never sent to the processor.
//
// Layout is by rectangles, so the spacing is the same everywhere: panels have a padding of Pad, rows are RowH high with Gap between them, columns
// are ColGap apart, and every text is centred vertically in its row. Typical use inside OnGUI:
//   Rect inner = UiTheme.Panel(new Rect(x, y, w, UiTheme.PanelHeight(3)));
//   Rect[] c = UiTheme.Cols(UiTheme.NextRow(ref inner), 0f, 120f);      // 0 = takes the rest, otherwise a width in pixels
//   GUI.Label(c[0], "text", UiTheme.Label);  if (UiTheme.Press(c[1], "Go")) ...
public static class UiTheme
{
    public enum Tone { Neutral, Ok, Warn, Bad, Info }

    // The whole UI is drawn in a virtual screen that is UiTheme.W x UiTheme.H big and scaled to the window, so the panels keep their proportions
    // when the window is resized or goes fullscreen. Every OnGUI that draws panels brackets its drawing with Begin() and End() and uses W and H
    // where it would use Screen.width and Screen.height.
    public static float Scale { get { return Mathf.Clamp(Screen.height / 1000f, 0.7f, 2.5f); } }
    public static float W { get { return Screen.width / Scale; } }
    public static float H { get { return Screen.height / Scale; } }
    public static void Begin() { float k = Scale; GUI.matrix = Matrix4x4.Scale(new Vector3(k, k, 1f)); }
    public static void End() { GUI.matrix = Matrix4x4.identity; }

    public const float Pad = 12f, RowH = 24f, Gap = 6f, ColGap = 8f;

    public static readonly Color PanelFill = new Color32(12, 17, 23, 232);
    public static readonly Color Line = new Color32(40, 52, 68, 255);
    public static readonly Color Accent = new Color32(80, 200, 232, 255);
    public static readonly Color TextMain = new Color32(238, 243, 249, 255);
    public static readonly Color TextMuted = new Color32(180, 193, 211, 255);
    public static readonly Color Green = new Color32(74, 222, 128, 255);
    public static readonly Color Amber = new Color32(251, 191, 36, 255);
    public static readonly Color Red = new Color32(248, 113, 113, 255);

    public static GUIStyle Name, Label, Muted, Value, Right, Button, ButtonLeft, Wrap;
    static GUIStyle[] statusStyles;
    static bool built;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics() { built = false; }

    public static void Ensure()
    {
        if (built && Button != null && Button.normal.background != null) return;
        Build();
    }

    static Texture2D Solid(Color c)
    {
        var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        t.hideFlags = HideFlags.HideAndDontSave;
        t.SetPixel(0, 0, c);
        t.Apply();
        return t;
    }

    static GUIStyle Text(int size, Color color, TextAnchor anchor, Font font = null)
    {
        var s = new GUIStyle();
        s.fontSize = size;
        s.alignment = anchor;
        s.normal.textColor = color;
        s.clipping = TextClipping.Clip;
        s.padding = new RectOffset(0, 0, 0, 0);
        s.margin = new RectOffset(0, 0, 0, 0);
        if (font != null) s.font = font;
        return s;
    }

    static GUIStyle Buttonish(TextAnchor anchor, int padLeft)
    {
        GUIStyle s = Text(12, TextMain, anchor);
        s.normal.background = Solid(new Color32(26, 35, 46, 255));
        s.hover.background = Solid(new Color32(36, 52, 72, 255));
        s.active.background = Solid(new Color32(14, 80, 98, 255));
        s.focused.background = s.normal.background;
        s.hover.textColor = Color.white;
        s.active.textColor = Color.white;
        s.focused.textColor = TextMain;
        s.padding = new RectOffset(padLeft, 0, 0, 0);
        return s;
    }

    static void Build()
    {
        Font mono = Font.CreateDynamicFontFromOSFont(new[] { "Consolas", "Menlo", "DejaVu Sans Mono", "Courier New" }, 13);
        Name = Text(14, TextMain, TextAnchor.MiddleLeft);
        Label = Text(12, TextMain, TextAnchor.MiddleLeft);
        Muted = Text(12, TextMuted, TextAnchor.MiddleLeft);
        Right = Text(11, TextMuted, TextAnchor.MiddleRight);
        Value = Text(13, TextMain, TextAnchor.MiddleLeft, mono);
        Wrap = Text(11, TextMuted, TextAnchor.UpperLeft);
        Wrap.wordWrap = true;
        Button = Buttonish(TextAnchor.MiddleCenter, 0);
        ButtonLeft = Buttonish(TextAnchor.MiddleLeft, 10);

        statusStyles = new GUIStyle[5];
        Color[] tone = { TextMuted, Green, Amber, Red, Accent };
        for (int i = 0; i < 5; i++) statusStyles[i] = Text(12, tone[i], TextAnchor.MiddleLeft);
        built = true;
    }

    public static Color ToneColor(Tone t)
    {
        switch (t)
        {
            case Tone.Ok: return Green;
            case Tone.Warn: return Amber;
            case Tone.Bad: return Red;
            case Tone.Info: return Accent;
            default: return TextMuted;
        }
    }

    // ---------------------------------------------------------------- drawing

    public static void Fill(Rect r, Color c)
    {
        Color old = GUI.color;
        GUI.color = c;
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = old;
    }

    public static void Separator(Rect r) { Fill(new Rect(r.x, r.y, r.width, 1f), Line); }

    // A panel: dark fill with a thin outline. Returns the area inside the padding.
    public static Rect Panel(Rect r)
    {
        Ensure();
        Fill(r, PanelFill);
        Fill(new Rect(r.x, r.y, r.width, 1f), Line);
        Fill(new Rect(r.x, r.yMax - 1f, r.width, 1f), Line);
        Fill(new Rect(r.x, r.y, 1f, r.height), Line);
        Fill(new Rect(r.xMax - 1f, r.y, 1f, r.height), Line);
        return new Rect(r.x + Pad, r.y + Pad, r.width - 2f * Pad, r.height - 2f * Pad);
    }

    // height of a panel that holds exactly this many rows
    public static float PanelHeight(int rows) { return 2f * Pad + rows * RowH + Mathf.Max(0, rows - 1) * Gap; }

    // the next row of 'inner' (the rest shrinks); h = 0 gives the standard row height
    public static Rect NextRow(ref Rect inner, float h = 0f)
    {
        if (h <= 0f) h = RowH;
        var row = new Rect(inner.x, inner.y, inner.width, h);
        inner.y += h + Gap;
        inner.height -= h + Gap;
        return row;
    }

    // columns of a row: a width above 0 is that many pixels, 0 shares what is left
    public static Rect[] Cols(Rect row, params float[] widths)
    {
        float fixedSum = 0f;
        int flex = 0;
        foreach (float w in widths) { if (w > 0f) fixedSum += w; else flex++; }
        float free = row.width - fixedSum - ColGap * (widths.Length - 1);
        float flexW = flex > 0 ? Mathf.Max(0f, free / flex) : 0f;
        var res = new Rect[widths.Length];
        float x = row.x;
        for (int i = 0; i < widths.Length; i++)
        {
            float w = widths[i] > 0f ? widths[i] : flexW;
            res[i] = new Rect(x, row.y, w, row.height);
            x += w + ColGap;
        }
        return res;
    }

    public static bool Press(Rect r, string text, bool left = false) { return GUI.Button(r, text, left ? ButtonLeft : Button); }

    // a small square in the colour of the tone and the text next to it, in the same colour
    public static void Status(Rect r, string text, Tone tone)
    {
        Ensure();
        Fill(new Rect(r.x, r.y + (r.height - 8f) * 0.5f, 8f, 8f), ToneColor(tone));
        GUI.Label(new Rect(r.x + 16f, r.y, Mathf.Max(0f, r.width - 16f), r.height), text, statusStyles[(int)tone]);
    }
}