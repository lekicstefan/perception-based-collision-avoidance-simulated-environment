using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

// The dashboard (spec 5.9): what the system is doing, live. It shows the TELEMETRY message of the processor (docs/protocol.md 5.3, up to 10 Hz)
// and the ego speed of the car. Debug UI only: it never sends anything and nothing here is used for control.
// Owned and drawn by ViewControls (key Y, hidden with H), so every scenario scene has it without any setup.
//
//   Link      the connection: waiting for the processor / connected (and how old the newest data is, in simulation time) / no data
//   Speed     the car's speed on a 0-100 km/h bar, with the comfort cap and the safety cap as markers
//   Caps      the two caps in km/h
//   Brake     the braking request and its regime (comfort, progressive, hard, emergency)
//   Risk      the risk level and the view it comes from (sight distance, unknown share)
//   Limit     what sets the cap: a track, or a virtual hazard behind an occlusion
//   Braking   what causes the braking, when there is any
//   Objects   tracks and virtual hazards considered, collision probability
//   Timing    latency of the command and the processor's stage times
// DrawOverlay (key O) puts the same tracks as boxes on the picture of the camera that is watched.
// DrawMap draws the minimap below it (key N): a top-down view in the car's frame with the tracks of the processor (outline and fill by state, an
// arrow for the ones that move, their id), the virtual hazards of occlusions (red bar across the path) and, for comparison, the ground truth
// (dim outlines of every Hazard object). The tracks are in the frame of the scan they come from, so they are moved back by the distance the car
// has driven since (speed x age of the data); the ground truth is drawn at the car's present position.
public class Dashboard
{
    JObject tel;
    string lastJson;

    const float BarMaxKmh = 100f;
    static readonly Color Track = new Color32(26, 35, 46, 255);

    public const float Width = 420f;
    public static float Height { get { return UiTheme.PanelHeight(9); } }

    // the tracks and virtual hazards of the last message, for the minimap
    class MapTrack { public int id; public string state; public bool isStatic; public float x, y, vx, vy, speed; public Vector2[] corners; }
    class MapVirtual { public float s; public string kind; }
    readonly List<MapTrack> mapTracks = new List<MapTrack>();
    readonly List<MapVirtual> mapVirtual = new List<MapVirtual>();

    void Refresh(NetworkHost net)
    {
        string json = net != null ? net.LatestTelemetry : null;
        if (ReferenceEquals(json, lastJson)) return;
        lastJson = json;
        if (json == null) { tel = null; return; }
        try { tel = JObject.Parse(json); }
        catch (System.Exception) { tel = null; }
        ReadMap();
    }

    void ReadMap()
    {
        mapTracks.Clear();
        mapVirtual.Clear();
        if (tel == null) return;
        if (tel.SelectToken("tracks") is JArray ta)
            foreach (JToken t in ta)
            {
                var m = new MapTrack
                {
                    id = (int)(Num(t, "id") ?? 0f),
                    state = Str(t, "state") ?? "",
                    isStatic = Flag(t, "static"),
                    x = Num(t, "x") ?? 0f,
                    y = Num(t, "y") ?? 0f,
                    vx = Num(t, "vx") ?? 0f,
                    vy = Num(t, "vy") ?? 0f,
                    speed = Num(t, "speed") ?? 0f
                };
                if (t.SelectToken("corners") is JArray ca && ca.Count >= 3)
                {
                    m.corners = new Vector2[ca.Count];
                    for (int i = 0; i < ca.Count; i++) m.corners[i] = new Vector2(ca[i][0].Value<float>(), ca[i][1].Value<float>());
                }
                mapTracks.Add(m);
            }
        if (tel.SelectToken("virtual") is JArray va)
            foreach (JToken v in va)
                mapVirtual.Add(new MapVirtual { s = Num(v, "s_m") ?? 0f, kind = Str(v, "kind") ?? "" });
    }

    // ---------------------------------------------------------------- reading the message

    static float? Num(JToken root, string path)
    {
        JToken t = root == null ? null : root.SelectToken(path);
        if (t == null || t.Type == JTokenType.Null) return null;
        return t.Value<float?>();
    }

    static string Str(JToken root, string path)
    {
        JToken t = root == null ? null : root.SelectToken(path);
        return t == null || t.Type == JTokenType.Null ? null : t.Value<string>();
    }

    static bool Flag(JToken root, string path)
    {
        JToken t = root == null ? null : root.SelectToken(path);
        return t != null && t.Type == JTokenType.Boolean && t.Value<bool>();
    }

    static string Kmh(float? mps)
    {
        if (!mps.HasValue) return "-";
        float k = mps.Value * 3.6f;
        return k >= 150f ? "no limit" : k.ToString("F0") + " km/h";
    }

    static string CauseText(JToken cause)
    {
        string kind = Str(cause, "kind");
        if (kind == null || kind == "none") return "-";
        string s = Str(cause, "label") ?? kind;
        string state = Str(cause, "state");
        if (!string.IsNullOrEmpty(state) && state != "virtual") s += " (" + state + ")";
        float? d = Num(cause, "d_path_m");
        if (d.HasValue) s += ", " + d.Value.ToString("F1") + " m ahead";
        float? cap = Num(cause, "cap_mps");
        if (cap.HasValue && cap.Value * 3.6f < 150f) s += ", cap " + Kmh(cap);
        return s;
    }

    // ---------------------------------------------------------------- drawing

    static Rect Line(ref Rect inner, string label)
    {
        Rect[] c = UiTheme.Cols(UiTheme.NextRow(ref inner), 80f, 0f);
        GUI.Label(c[0], label, UiTheme.Muted);
        return c[1];
    }

    public void Draw(Rect panel, Vehicle vehicle, NetworkHost net)
    {
        Refresh(net);
        Rect inner = UiTheme.Panel(panel);

        // link
        Rect r = Line(ref inner, "Link");
        if (net == null || !net.waitForProcessor) UiTheme.Status(r, "No processor link in this scene", UiTheme.Tone.Neutral);
        else if (!net.Started) UiTheme.Status(r, "Waiting for the processor", UiTheme.Tone.Warn);
        else if (tel == null) UiTheme.Status(r, "Connected, no telemetry yet", UiTheme.Tone.Warn);
        else
        {
            float? t = Num(tel, "t");
            float lag = t.HasValue ? (float)(SimClock.Now - t.Value) : 0f;
            if (lag < 0.6f) UiTheme.Status(r, "Connected, data " + (lag * 1000f).ToString("F0") + " ms old", UiTheme.Tone.Ok);
            else UiTheme.Status(r, "No data for " + lag.ToString("F1") + " s", UiTheme.Tone.Bad);
        }

        bool have = tel != null;
        float? comfort = have ? Num(tel, "command.comfort_cap") : null;
        float? safety = have ? Num(tel, "command.safety_cap") : null;

        // speed bar with the two caps as markers
        float kmh = vehicle != null ? vehicle.Speed * 3.6f : 0f;
        Rect[] c = UiTheme.Cols(Line(ref inner, "Speed"), 84f, 0f);
        GUI.Label(c[0], kmh.ToString("F0") + " km/h", UiTheme.Value);
        Rect bar = new Rect(c[1].x, c[1].y + 7f, c[1].width, 10f);
        UiTheme.Fill(bar, Track);
        UiTheme.Fill(new Rect(bar.x, bar.y, bar.width * Mathf.Clamp01(kmh / BarMaxKmh), bar.height), UiTheme.Accent);
        Marker(bar, comfort, UiTheme.Amber);
        Marker(bar, safety, UiTheme.Red);

        // caps
        c = UiTheme.Cols(Line(ref inner, "Caps"), 0f, 0f);
        UiTheme.Status(c[0], "comfort " + Kmh(comfort), UiTheme.Tone.Warn);
        UiTheme.Status(c[1], "safety " + Kmh(safety), UiTheme.Tone.Bad);

        // brake request and regime
        c = UiTheme.Cols(Line(ref inner, "Brake"), 0f, 150f);
        float? brake = have ? Num(tel, "command.brake_request") : null;
        GUI.Label(c[0], brake.HasValue ? brake.Value.ToString("F1") + " m/s²" : "-", UiTheme.Value);
        int regime = have ? (int)(Num(tel, "decision.regime") ?? 0f) : 0;
        bool emergency = have && Flag(tel, "command.emergency");
        if (!have) UiTheme.Status(c[1], "-", UiTheme.Tone.Neutral);
        else if (emergency || regime == 4) UiTheme.Status(c[1], "Emergency", UiTheme.Tone.Bad);
        else if (regime == 3) UiTheme.Status(c[1], "Hard", UiTheme.Tone.Warn);
        else if (regime == 2) UiTheme.Status(c[1], "Progressive", UiTheme.Tone.Warn);
        else UiTheme.Status(c[1], "Comfort", UiTheme.Tone.Ok);

        // risk level and the view behind it
        c = UiTheme.Cols(Line(ref inner, "Risk"), 100f, 0f);
        int level = have ? (int)(Num(tel, "risk.level") ?? 0f) : -1;
        if (level < 0) UiTheme.Status(c[0], "-", UiTheme.Tone.Neutral);
        else if (level == 0) UiTheme.Status(c[0], "Low", UiTheme.Tone.Ok);
        else if (level == 1) UiTheme.Status(c[0], "Medium", UiTheme.Tone.Warn);
        else UiTheme.Status(c[0], "High", UiTheme.Tone.Bad);
        if (have)
        {
            float? sight = Num(tel, "risk.sight_distance_m"), unknown = Num(tel, "risk.unknown_fraction");
            GUI.Label(c[1], "sight " + (sight.HasValue ? sight.Value.ToString("F0") + " m" : "-") + "   unknown " + (unknown.HasValue ? (unknown.Value * 100f).ToString("F0") + " %" : "-"), UiTheme.Muted);
        }

        // causes
        GUI.Label(Line(ref inner, "Limit"), have ? CauseText(tel.SelectToken("decision.comfort_cause")) : "-", UiTheme.Label);
        GUI.Label(Line(ref inner, "Braking"), have ? CauseText(tel.SelectToken("decision.brake_cause")) : "-", UiTheme.Label);

        // objects
        string objects = "-";
        if (have)
        {
            int tracks = tel.SelectToken("tracks") is JArray ta ? ta.Count : 0;
            int virt = (int)(Num(tel, "decision.n_virtual") ?? 0f);
            float? p = Num(tel, "collision.p");
            objects = tracks + " tracks   " + virt + " virtual   collision p " + (p.HasValue ? p.Value.ToString("0.00") : "-");
        }
        GUI.Label(Line(ref inner, "Objects"), objects, UiTheme.Label);

        // timing
        string timing = "-";
        if (have)
        {
            float? lat = Num(tel, "timing.latency_s"), total = Num(tel, "timing.stage_ms.total"), track = Num(tel, "timing.perception_tracking_ms");
            timing = "latency " + (lat.HasValue ? (lat.Value * 1000f).ToString("F0") : "-") + " ms   risk stage " + (total.HasValue ? total.Value.ToString("F1") : "-") +
                     " ms   tracking " + (track.HasValue ? track.Value.ToString("F1") : "-") + " ms";
        }
        GUI.Label(Line(ref inner, "Timing"), timing, UiTheme.Label);
    }


    // ---------------------------------------------------------------- minimap

    public const float MapSize = 290f;
    public static float MapHeight { get { return 2f * UiTheme.Pad + MapSize + UiTheme.Gap + UiTheme.RowH; } }

    const float MapForwardM = 44f, MapRearM = 6f;
    static readonly Color ColTruth = new Color32(96, 116, 144, 255);
    static readonly Color ColTentative = new Color32(154, 168, 188, 255);
    static readonly Color ColVirtual = new Color32(248, 113, 113, 255);

    static Material glMat;
    static GUIStyle idStyle, gridStyle;
    Vector2[] egoRect;
    Rect map;
    float pxPerM;
    Vector2 origin;
    readonly List<Vector2> fillV = new List<Vector2>();
    readonly List<Color> fillC = new List<Color>();
    readonly List<Vector2> lineV = new List<Vector2>();
    readonly List<Color> lineC = new List<Color>();

    Vector2 Px(float x, float y) { return new Vector2(origin.x - y * pxPerM, origin.y - x * pxPerM); }

    public void DrawMap(Rect panel, Vehicle vehicle, NetworkHost net)
    {
        Refresh(net);
        Rect inner = UiTheme.Panel(panel);
        map = UiTheme.NextRow(ref inner, MapSize);
        Rect legend = UiTheme.NextRow(ref inner);
        UiTheme.Fill(map, new Color32(8, 12, 17, 255));

        if (vehicle != null && Event.current.type == EventType.Repaint && EnsureMaterial())
        {
            pxPerM = map.height / (MapForwardM + MapRearM);
            origin = new Vector2(map.center.x, map.yMax - MapRearM * pxPerM);
            fillV.Clear(); fillC.Clear(); lineV.Clear(); lineC.Clear();

            for (float d = 10f; d <= MapForwardM; d += 10f)                      // distance lines
                AddLine(Px(d, -100f), Px(d, 100f), new Color32(24, 33, 44, 255));

            float lag = 0f;
            if (tel != null) { float? t = Num(tel, "t"); if (t.HasValue) lag = Mathf.Clamp((float)(SimClock.Now - t.Value), 0f, 1f); }
            float shift = vehicle.Speed * lag;

            // ground truth: every Hazard object as a dim outline, in the car's present frame
            Transform vt = vehicle.transform;
            foreach (Hazard h in Hazard.All)
            {
                if (h == null || h.Length <= 0f || h.Width <= 0f) continue;
                Vector3 c = h.WorldCenter;
                if ((c - vt.position).sqrMagnitude > 90f * 90f) continue;
                Vector3 r = h.transform.right, f = h.transform.forward;
                r.y = 0f; f.y = 0f;
                var poly = new Vector2[4];
                for (int i = 0; i < 4; i++)
                {
                    Vector3 w = c + r * (h.Width * 0.5f * ((i == 0 || i == 3) ? -1f : 1f)) + f * (h.Length * 0.5f * (i < 2 ? -1f : 1f));
                    Vector3 l = vt.InverseTransformPoint(w);
                    poly[i] = Px(l.z, -l.x);
                }
                AddOutline(poly, ColTruth);
            }

            // virtual hazards: a bar across the path at the distance they act
            foreach (MapVirtual v in mapVirtual)
            {
                float x = v.s - shift;
                AddLine(Px(x, -3f), Px(x, 3f), ColVirtual);
                AddLine(Px(x - 0.4f, -3f), Px(x - 0.4f, 3f), ColVirtual);
            }

            // tracks
            foreach (MapTrack m in mapTracks)
            {
                Color col = m.state == "confirmed" ? UiTheme.Accent : m.state == "coasting" ? UiTheme.Amber : ColTentative;
                Vector2[] poly;
                if (m.corners != null)
                {
                    poly = new Vector2[m.corners.Length];
                    for (int i = 0; i < poly.Length; i++) poly[i] = Px(m.corners[i].x - shift, m.corners[i].y);
                }
                else
                    poly = new[] { Px(m.x - shift - 0.5f, m.y - 0.5f), Px(m.x - shift - 0.5f, m.y + 0.5f), Px(m.x - shift + 0.5f, m.y + 0.5f), Px(m.x - shift + 0.5f, m.y - 0.5f) };
                Color fill = col; fill.a = 0.28f;
                AddFill(poly, fill);
                AddOutline(poly, col);
                if (!m.isStatic && m.speed > 0.5f)                               // moving: an arrow of one second
                    AddLine(Px(m.x - shift, m.y), Px(m.x - shift + m.vx, m.y + m.vy), col);
            }

            // the car
            Vector2[] ego = EgoPoly(vehicle);
            AddFill(ego, new Color(0.9f, 0.93f, 0.97f, 0.9f));
            AddLine(Px(0f, 0f), Px(MapForwardM, 0f), new Color(0.9f, 0.93f, 0.97f, 0.25f));
            Flush();
        }

        // text on the map: distance marks and track ids
        EnsureStyles();
        if (vehicle != null && Event.current.type == EventType.Repaint)
        {
            for (float d = 10f; d <= MapForwardM; d += 10f)
                GUI.Label(new Rect(map.x + 4f, origin.y - d * pxPerM - 14f, 50f, 12f), d.ToString("F0") + " m", gridStyle);
            float lag = 0f;
            if (tel != null) { float? t = Num(tel, "t"); if (t.HasValue) lag = Mathf.Clamp((float)(SimClock.Now - t.Value), 0f, 1f); }
            float shift = vehicle.Speed * lag;
            foreach (MapTrack m in mapTracks)
            {
                Vector2 p = Px(m.x - shift, m.y);
                if (p.x < map.x + 6f || p.x > map.xMax - 6f || p.y < map.y + 6f || p.y > map.yMax - 6f) continue;
                GUI.Label(new Rect(p.x - 12f, p.y - 22f, 24f, 12f), m.id.ToString(), idStyle);
            }
        }

        // legend
        Rect[] lc = UiTheme.Cols(legend, 0f, 0f, 0f, 0f, 0f);
        Key(lc[0], "confirmed", UiTheme.Accent);
        Key(lc[1], "tentative", ColTentative);
        Key(lc[2], "coasting", UiTheme.Amber);
        Key(lc[3], "virtual", ColVirtual);
        Key(lc[4], "truth", ColTruth);
    }

    static void EnsureStyles()
    {
        if (gridStyle != null) return;
        gridStyle = new GUIStyle(UiTheme.Muted) { fontSize = 10 };
        idStyle = new GUIStyle(UiTheme.Muted) { fontSize = 10, alignment = TextAnchor.MiddleCenter };
        idStyle.normal.textColor = UiTheme.TextMain;
    }

    // ---------------------------------------------------------------- tracks on the camera picture

    const float OverlayHeightM = 1.6f;

    // Draws the tracks and virtual hazards of the last message as boxes (footprint, 1.6 m high) on the picture of the camera that is watched.
    // The positions are in the car's frame of the scan, so they are moved back by speed x age of the data, like on the minimap, and placed with
    // the car's present transform. The box is a guide to where the processor thinks the object is, not a measurement of its height.
    public void DrawOverlay(Camera cam, Vehicle vehicle, NetworkHost net)
    {
        if (cam == null || vehicle == null || Event.current.type != EventType.Repaint || !EnsureMaterial()) return;
        Refresh(net);
        if (mapTracks.Count == 0 && mapVirtual.Count == 0) return;
        EnsureStyles();

        float lag = 0f;
        if (tel != null) { float? t = Num(tel, "t"); if (t.HasValue) lag = Mathf.Clamp((float)(SimClock.Now - t.Value), 0f, 1f); }
        float shift = vehicle.Speed * lag;

        map = new Rect(0f, 0f, UiTheme.W, UiTheme.H);                           // lines are clipped to the screen
        fillV.Clear(); fillC.Clear(); lineV.Clear(); lineC.Clear();
        Transform vt = vehicle.transform;
        Vector3 up = vt.up * OverlayHeightM;
        var labels = new List<KeyValuePair<Vector3, string>>();

        foreach (MapTrack m in mapTracks)
        {
            Color col = m.state == "confirmed" ? UiTheme.Accent : m.state == "coasting" ? UiTheme.Amber : ColTentative;
            Vector2[] c = m.corners ?? new[] { new Vector2(m.x - 0.5f, m.y - 0.5f), new Vector2(m.x - 0.5f, m.y + 0.5f), new Vector2(m.x + 0.5f, m.y + 0.5f), new Vector2(m.x + 0.5f, m.y - 0.5f) };
            var low = new Vector3[c.Length];
            Vector3 top = Vector3.zero;
            for (int i = 0; i < c.Length; i++)
            {
                low[i] = vt.TransformPoint(Frames.ToUnity(c[i].x - shift, c[i].y, 0f));
                top += low[i] + up;
            }
            for (int i = 0; i < c.Length; i++)
            {
                int j = (i + 1) % c.Length;
                Edge(cam, low[i], low[j], col);
                Edge(cam, low[i] + up, low[j] + up, col);
                Edge(cam, low[i], low[i] + up, col);
            }
            string text = m.id + " " + m.state + (!m.isStatic && m.speed > 0.5f ? " " + (m.speed * 3.6f).ToString("F0") + " km/h" : "");
            labels.Add(new KeyValuePair<Vector3, string>(top / c.Length, text));
        }

        foreach (MapVirtual v in mapVirtual)                                    // a gate across the path where the virtual hazard acts
        {
            Vector3 a = vt.TransformPoint(Frames.ToUnity(v.s - shift, -1.8f, 0f)), b = vt.TransformPoint(Frames.ToUnity(v.s - shift, 1.8f, 0f));
            Edge(cam, a, b, ColVirtual);
            Edge(cam, a + up, b + up, ColVirtual);
            Edge(cam, a, a + up, ColVirtual);
            Edge(cam, b, b + up, ColVirtual);
            labels.Add(new KeyValuePair<Vector3, string>((a + b) * 0.5f + up, "virtual " + v.kind));
        }
        Flush();

        Vector3 f = cam.transform.forward, o = cam.transform.position;
        foreach (KeyValuePair<Vector3, string> l in labels)
        {
            if (Vector3.Dot(l.Key - o, f) < 1f) continue;
            Vector2 p = ToGui(cam, l.Key);
            if (p.x < 0f || p.x > UiTheme.W || p.y < 16f || p.y > UiTheme.H) continue;
            Vector2 sz = idStyle.CalcSize(new GUIContent(l.Value));
            var r = new Rect(p.x - sz.x * 0.5f - 4f, p.y - 18f, sz.x + 8f, 16f);
            UiTheme.Fill(r, UiTheme.PanelFill);
            GUI.Label(r, l.Value, idStyle);
        }
    }

    // one edge in 3D: cut at the camera's near side, projected, drawn two pixels thick
    void Edge(Camera cam, Vector3 a, Vector3 b, Color c)
    {
        Vector3 f = cam.transform.forward, o = cam.transform.position;
        float da = Vector3.Dot(a - o, f) - 0.2f, db = Vector3.Dot(b - o, f) - 0.2f;
        if (da < 0f && db < 0f) return;
        if (da < 0f) a = Vector3.Lerp(a, b, da / (da - db));
        else if (db < 0f) b = Vector3.Lerp(a, b, da / (da - db));
        Vector2 pa = ToGui(cam, a), pb = ToGui(cam, b);
        AddLine(pa, pb, c);
        AddLine(pa + Vector2.up, pb + Vector2.up, c);
    }

    // screen position in the units of the scaled UI (y down)
    static Vector2 ToGui(Camera cam, Vector3 world)
    {
        Vector3 s = cam.WorldToScreenPoint(world);
        float k = UiTheme.Scale;
        return new Vector2(s.x / k, (Screen.height - s.y) / k);
    }

    static void Key(Rect r, string text, Color color)
    {
        UiTheme.Fill(new Rect(r.x, r.y + (r.height - 8f) * 0.5f, 8f, 8f), color);
        GUI.Label(new Rect(r.x + 14f, r.y, Mathf.Max(0f, r.width - 14f), r.height), text, UiTheme.Muted);
    }

    Vector2[] EgoPoly(Vehicle v)
    {
        if (egoRect == null)
        {
            float x0 = -1f, x1 = 3.8f, hw = 0.95f, yc = 0f;
            BoxCollider bc = v.GetComponentInChildren<BoxCollider>();
            if (bc != null)
            {
                Vector3 c = v.transform.InverseTransformPoint(bc.transform.TransformPoint(bc.center));
                Vector3 s = Vector3.Scale(bc.size, bc.transform.lossyScale);
                x0 = c.z - s.z * 0.5f; x1 = c.z + s.z * 0.5f; hw = Mathf.Abs(s.x) * 0.5f; yc = -c.x;
            }
            egoRect = new[] { new Vector2(x0, yc - hw), new Vector2(x0, yc + hw), new Vector2(x1, yc + hw), new Vector2(x1, yc - hw) };
        }
        var p = new Vector2[4];
        for (int i = 0; i < 4; i++) p[i] = Px(egoRect[i].x, egoRect[i].y);
        return p;
    }

    // ---- drawing with GL in pixel coordinates (y down, like the GUI), clipped to the map

    static bool EnsureMaterial()
    {
        if (glMat != null) return true;
        Shader sh = Shader.Find("Hidden/Internal-Colored");
        if (sh == null) return false;
        glMat = new Material(sh) { hideFlags = HideFlags.HideAndDontSave };
        glMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        glMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        glMat.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        glMat.SetInt("_ZWrite", 0);
        glMat.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
        return true;
    }

    void AddLine(Vector2 a, Vector2 b, Color c)
    {
        if (!ClipSegment(ref a, ref b, map)) return;
        lineV.Add(a); lineV.Add(b); lineC.Add(c); lineC.Add(c);
    }

    void AddOutline(Vector2[] poly, Color c)
    {
        for (int i = 0; i < poly.Length; i++) AddLine(poly[i], poly[(i + 1) % poly.Length], c);
    }

    void AddFill(Vector2[] poly, Color c)
    {
        List<Vector2> p = ClipPolygon(poly, map);
        for (int i = 1; i + 1 < p.Count; i++)
        {
            fillV.Add(p[0]); fillV.Add(p[i]); fillV.Add(p[i + 1]);
            fillC.Add(c); fillC.Add(c); fillC.Add(c);
        }
    }

    void Flush()
    {
        GL.PushMatrix();
        GL.LoadPixelMatrix(0f, Screen.width, Screen.height, 0f);
        GL.MultMatrix(Matrix4x4.Scale(new Vector3(UiTheme.Scale, UiTheme.Scale, 1f)));      // the map is drawn in the scaled UI coordinates
        glMat.SetPass(0);
        GL.Begin(GL.TRIANGLES);
        for (int i = 0; i < fillV.Count; i++) { GL.Color(fillC[i]); GL.Vertex3(fillV[i].x, fillV[i].y, 0f); }
        GL.End();
        GL.Begin(GL.LINES);
        for (int i = 0; i < lineV.Count; i++) { GL.Color(lineC[i]); GL.Vertex3(lineV[i].x, lineV[i].y, 0f); }
        GL.End();
        GL.PopMatrix();
    }

    // Liang-Barsky: the part of the segment inside the rectangle
    static bool ClipSegment(ref Vector2 a, ref Vector2 b, Rect r)
    {
        float t0 = 0f, t1 = 1f, dx = b.x - a.x, dy = b.y - a.y;
        float[] p = { -dx, dx, -dy, dy };
        float[] q = { a.x - r.xMin, r.xMax - a.x, a.y - r.yMin, r.yMax - a.y };
        for (int i = 0; i < 4; i++)
        {
            if (Mathf.Approximately(p[i], 0f)) { if (q[i] < 0f) return false; continue; }
            float t = q[i] / p[i];
            if (p[i] < 0f) { if (t > t1) return false; if (t > t0) t0 = t; }
            else { if (t < t0) return false; if (t < t1) t1 = t; }
        }
        Vector2 na = new Vector2(a.x + t0 * dx, a.y + t0 * dy), nb = new Vector2(a.x + t1 * dx, a.y + t1 * dy);
        a = na; b = nb;
        return true;
    }

    // Sutherland-Hodgman against the four sides of the rectangle
    static List<Vector2> ClipPolygon(Vector2[] poly, Rect r)
    {
        var cur = new List<Vector2>(poly);
        for (int side = 0; side < 4 && cur.Count > 0; side++)
        {
            var next = new List<Vector2>();
            for (int i = 0; i < cur.Count; i++)
            {
                Vector2 a = cur[i], b = cur[(i + 1) % cur.Count];
                bool ina = Inside(a, r, side), inb = Inside(b, r, side);
                if (ina) next.Add(a);
                if (ina != inb) next.Add(Cross(a, b, r, side));
            }
            cur = next;
        }
        return cur;
    }

    static bool Inside(Vector2 p, Rect r, int side)
    {
        switch (side) { case 0: return p.x >= r.xMin; case 1: return p.x <= r.xMax; case 2: return p.y >= r.yMin; default: return p.y <= r.yMax; }
    }

    static Vector2 Cross(Vector2 a, Vector2 b, Rect r, int side)
    {
        float t;
        switch (side)
        {
            case 0: t = (r.xMin - a.x) / (b.x - a.x); break;
            case 1: t = (r.xMax - a.x) / (b.x - a.x); break;
            case 2: t = (r.yMin - a.y) / (b.y - a.y); break;
            default: t = (r.yMax - a.y) / (b.y - a.y); break;
        }
        return a + (b - a) * t;
    }

    static void Marker(Rect bar, float? capMps, Color color)
    {
        if (!capMps.HasValue) return;
        float k = capMps.Value * 3.6f;
        if (k >= BarMaxKmh) return;
        float x = bar.x + bar.width * Mathf.Clamp01(k / BarMaxKmh);
        UiTheme.Fill(new Rect(x - 1f, bar.y - 4f, 2f, bar.height + 8f), color);
    }
}