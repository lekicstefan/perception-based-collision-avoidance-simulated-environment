using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

// What you see while a run is going: which camera, and which debug overlays. Everything here is for the person watching. None of it is
// sent to the processor and none of it is drawn into the sensor camera (which renders to a texture), so a run looks the same to the system
// whatever is switched on.
//
// Keys (click the Game view first; ignored while a text field has the keyboard):
//   1-9    camera (the cameras below the Vehicle, in name order)      W / Shift+W   next / previous camera
//   Q      camera view in the corner (the image the processor receives)
//   A      lidar view in the corner: off / range image / bird's-eye / both
//   S      lidar dots: off / objects / all returns
//   D      lidar beams: off / one row / grid / objects               F and G   move the beam row down / up (always works, used by "row")
//   Z      dashboard (the system's live state, from the processor's telemetry)
//   X      minimap under the dashboard (tracks, virtual hazards, ground truth)
//   C      tracks as boxes on the camera picture
//   V      the route                B   boxes around the Hazard objects (ground truth)
//   Tab    hide / show all panels and pictures (for screenshots)
//   Shift+Tab   hide / show the effects in the world (dots, beams, boxes, route, tracks); they stay switched on, they are only not drawn
// The choices are kept when the scene is restarted. Run-control keys (Esc, Space, E, R, arrows, 0) belong to the ScenarioManager.
[DefaultExecutionOrder(-250)]
public class ViewControls : MonoBehaviour
{
    public enum DotMode { Off, Objects, All }
    public enum BeamMode { Off, Ring, Sparse, Objects }
    public enum LidarViewMode { Off, RangeImage, BirdsEye, Both }

    [Header("Wiring (found in the scene when left empty)")]
    public Vehicle vehicle;
    public RouteDriver driver;
    public LidarSensor lidar;
    public CameraSensor cameraSensor;
    public NetworkHost network;
    [Tooltip("Parent of the view cameras. Empty: every camera below the Vehicle that renders to the screen.")]
    public Transform cameraRoot;

    [Header("Start values (first run of a session; restarts keep what you chose)")]
    public int startCamera = 0;
    public BeamMode startBeams = BeamMode.Off;
    public LidarViewMode startLidarView = LidarViewMode.Off;
    public bool startCameraView = true;

    [Header("Lidar beams")]
    public int sparseColumnStep = 15;
    public int sparseRowStep = 8;
    public int objectBeamStride = 6;
    public int maxBeams = 1500;

    [Header("Lidar view")]
    public float birdsEyeRangeM = 40f;

    public bool hotkeys = true;

    // the choices, kept across scene reloads
    static bool initialised;
    static int camIndex;
    static DotMode dots;
    static BeamMode beams;
    static LidarViewMode lidarView;
    static bool cameraView, groundTruthBoxes, routeLine;
    static bool dashboardOn, minimapOn, trackOverlayOn, effectsHidden;
    static bool hudVisible = true;
    static int ringRow = -1;

    public static bool HudVisible { get { return hudVisible; } }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics() { initialised = false; hudVisible = true; ringRow = -1; dashboardOn = false; minimapOn = false; trackOverlayOn = false; effectsHidden = false; }

    readonly List<Camera> cams = new List<Camera>();
    readonly Dictionary<Camera, int> ownDisplay = new Dictionary<Camera, int>();     // the Target Display each camera has in the scene
    readonly LidarOverlay overlay = new LidarOverlay();
    readonly LidarPanel panel = new LidarPanel();
    readonly Dashboard dashboard = new Dashboard();
    Material mat;
    bool materialFailed, statsDefault;
    Vector3[] routePoints;
    int lastBeamCount;

    // ---------------------------------------------------------------- setup

    void Awake()
    {
        if (!initialised)
        {
            camIndex = startCamera;
            dots = DotMode.Off;
            beams = startBeams;
            lidarView = startLidarView;
            cameraView = startCameraView;
            groundTruthBoxes = false;
            routeLine = false;
            initialised = true;
        }

        if (vehicle == null) vehicle = FindFirstObjectByType<Vehicle>();
        if (driver == null) driver = FindFirstObjectByType<RouteDriver>();
        if (lidar == null) lidar = FindFirstObjectByType<LidarSensor>();
        if (cameraSensor == null) cameraSensor = FindFirstObjectByType<CameraSensor>();
        if (network == null) network = FindFirstObjectByType<NetworkHost>();

        FindCameras();
        // the one-line stats of the sensors sat on top of the run panel and the dashboard shows the same numbers
        statsDefault = false;
        PoseEstimator pose = FindFirstObjectByType<PoseEstimator>();
        if (pose != null) pose.showStats = false;
    }

    void FindCameras()
    {
        cams.Clear();
        Transform root = cameraRoot != null ? cameraRoot : (vehicle != null ? vehicle.transform : null);
        Camera[] found = root != null ? root.GetComponentsInChildren<Camera>(true) : FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        foreach (Camera c in found)
            if (c.targetTexture == null && c.name != "SensorCamera") cams.Add(c);     // the sensor camera renders to a texture and is not a view
        cams.Sort((a, b) => string.CompareOrdinal(a.name, b.name));
        ownDisplay.Clear();
        foreach (Camera c in cams) ownDisplay[c] = c.targetDisplay;

        if (cams.Count == 0)
        {
            UnityEngine.Debug.LogWarning("VIEW CONTROLS: no camera found below the Vehicle (or the Camera Root), the camera keys do nothing");
            return;
        }
        camIndex = Mathf.Clamp(camIndex, 0, cams.Count - 1);
        ApplyCamera();
    }

    // Only the chosen camera renders, and only its audio listener is on (several of them produce a warning every frame).
    // The Game view shows Display 1, so the chosen camera is sent to Display 1 whatever its Target Display is in the scene (the cameras of the
    // car are set to Display 1 to 5 so that each can be watched in its own Game window); the others get their own display back.
    // A camera on an inactive object does not render whatever its own switch says, so the chosen camera's object and its parents are switched on first.
    void ApplyCamera()
    {
        if (cams.Count == 0) return;
        Camera target = cams[camIndex];
        for (Transform t = target.transform; t != null; t = t.parent)
            if (!t.gameObject.activeSelf)
            {
                UnityEngine.Debug.Log("VIEW CONTROLS: '" + t.name + "' was inactive, switched on so that the camera '" + target.name + "' can render");
                t.gameObject.SetActive(true);
            }
        for (int i = 0; i < cams.Count; i++)
        {
            bool on = i == camIndex;
            cams[i].targetDisplay = on ? 0 : ownDisplay[cams[i]];
            cams[i].enabled = on;
            AudioListener al = cams[i].GetComponent<AudioListener>();
            if (al != null) al.enabled = on;
        }
        UnityEngine.Debug.Log("VIEW | camera '" + target.name + "' (" + (camIndex + 1) + " of " + cams.Count + ") on Display 1 (its own: " + (ownDisplay[target] + 1) + ") | rendering: " + target.isActiveAndEnabled);
    }

    void OnEnable()
    {
        Camera.onPostRender += OnPostRenderCamera;                        // built-in render pipeline
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering; // scriptable render pipelines
    }

    void OnDisable()
    {
        Camera.onPostRender -= OnPostRenderCamera;
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
    }

    void OnDestroy()
    {
        if (mat != null) Destroy(mat);
    }

    // ---------------------------------------------------------------- actions

    void SelectCamera(int index)
    {
        if (cams.Count == 0) return;
        camIndex = ((index % cams.Count) + cams.Count) % cams.Count;
        ApplyCamera();
    }

    void NextCamera(int delta) { SelectCamera(camIndex + delta); }
    void CycleDots() { dots = (DotMode)(((int)dots + 1) % 3); }
    void CycleBeams() { beams = (BeamMode)(((int)beams + 1) % 4); }
    void CycleLidarView() { lidarView = (LidarViewMode)(((int)lidarView + 1) % 4); }
    void ToggleCameraView() { cameraView = !cameraView; }
    void ToggleBoxes() { groundTruthBoxes = !groundTruthBoxes; }
    void ToggleRoute() { routeLine = !routeLine; }
    void ToggleHud() { hudVisible = !hudVisible; }
    void ToggleEffects() { effectsHidden = !effectsHidden; }
    void ToggleDashboard() { dashboardOn = !dashboardOn; }
    void ToggleMinimap() { minimapOn = !minimapOn; }
    void ToggleTrackOverlay() { trackOverlayOn = !trackOverlayOn; }

    int Rows { get { return lidar != null && lidar.rig != null && lidar.rig.Config != null ? lidar.rig.Config.lidar.Rows : 0; } }

    // the beam row closest to horizontal, until the person picks another one
    int RingRow()
    {
        int rows = Rows;
        if (rows <= 0) return 0;
        if (ringRow < 0)
        {
            float[] el = lidar.rig.Config.lidar.ElevationsDeg;
            int best = 0;
            for (int r = 1; r < el.Length; r++)
                if (Mathf.Abs(el[r]) < Mathf.Abs(el[best])) best = r;
            ringRow = best;
        }
        ringRow = Mathf.Clamp(ringRow, 0, rows - 1);
        return ringRow;
    }

    void MoveRing(int delta)
    {
        int rows = Rows;
        if (rows <= 0) return;
        ringRow = Mathf.Clamp(RingRow() + delta, 0, rows - 1);
    }

    // ---------------------------------------------------------------- per frame

    void Update()
    {
        if (cameraSensor != null) cameraSensor.showPreview = cameraView && hudVisible;
        if (lidar != null)
        {
            lidar.showStats = hudVisible && statsDefault;
            bool need = dots != DotMode.Off || beams != BeamMode.Off || lidarView == LidarViewMode.BirdsEye || lidarView == LidarViewMode.Both;
            if (need) overlay.Refresh(lidar);
            if (hudVisible && lidarView != LidarViewMode.Off && vehicle != null)
                panel.Refresh(lidar, overlay, lidarView, vehicle.transform, birdsEyeRangeM);
        }
    }

    // ---------------------------------------------------------------- drawing into the player's camera

    void OnPostRenderCamera(Camera cam) { DrawOverlays(cam); }
    void OnEndCameraRendering(ScriptableRenderContext context, Camera cam) { DrawOverlays(cam); }

    bool EnsureMaterial()
    {
        if (mat != null) return true;
        if (materialFailed) return false;
        Shader s = Shader.Find("Hidden/Internal-Colored");
        if (s == null)
        {
            materialFailed = true;
            UnityEngine.Debug.LogError("VIEW CONTROLS: the shader Hidden/Internal-Colored was not found, the lidar overlays are off");
            return false;
        }
        mat = new Material(s);
        mat.hideFlags = HideFlags.HideAndDontSave;
        mat.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_Cull", (int)CullMode.Off);
        mat.SetInt("_ZWrite", 0);
        return true;
    }

    void DrawOverlays(Camera cam)
    {
        if (cam.targetTexture != null) return;       // sensor cameras render to textures: never draw into them
        if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;
        bool anyLidar = lidar != null && (dots != DotMode.Off || beams != BeamMode.Off) && overlay.Count > 0;
        if (effectsHidden) return;
        if (!anyLidar && !groundTruthBoxes && !routeLine) return;
        if (!EnsureMaterial()) return;

        GL.PushMatrix();
        if (anyLidar && dots != DotMode.Off)
        {
            mat.SetInt("_ZTest", (int)CompareFunction.LessEqual);
            mat.SetPass(0);
            overlay.DrawDots(cam, dots);
        }
        if (anyLidar && beams != BeamMode.Off)
        {
            mat.SetInt("_ZTest", (int)CompareFunction.LessEqual);
            mat.SetPass(0);
            lastBeamCount = overlay.DrawBeams(beams, RingRow(), sparseColumnStep, sparseRowStep, objectBeamStride, maxBeams);
        }
        if (groundTruthBoxes)
        {
            mat.SetInt("_ZTest", (int)CompareFunction.LessEqual);
            mat.SetPass(0);
            LidarOverlay.DrawHazardBoxes();
        }
        if (routeLine)
        {
            if (routePoints == null) BuildRoute();
            if (routePoints != null)
            {
                mat.SetInt("_ZTest", (int)CompareFunction.LessEqual);
                mat.SetPass(0);
                LidarOverlay.DrawPolyline(routePoints);
            }
        }
        GL.PopMatrix();
    }

    void BuildRoute()
    {
        if (driver == null || driver.Track == null) return;
        Route t = driver.Track;
        int n = Mathf.Max(2, Mathf.CeilToInt(t.Length) + 1);       // a point every metre
        routePoints = new Vector3[n];
        for (int i = 0; i < n; i++)
        {
            float s = Mathf.Min(t.Length, i * t.Length / (n - 1));
            routePoints[i] = t.Sample(s).position + Vector3.up * 0.1f;
        }
    }

    // ---------------------------------------------------------------- keys and HUD

    void HandleKey(Event e)
    {
        KeyCode k = e.keyCode;
        if (k >= KeyCode.Alpha1 && k <= KeyCode.Alpha9)
        {
            if (k - KeyCode.Alpha1 < cams.Count) SelectCamera(k - KeyCode.Alpha1);
            e.Use();
            return;
        }
        switch (k)
        {
            case KeyCode.W: NextCamera(e.shift ? -1 : 1); break;
            case KeyCode.Q: ToggleCameraView(); break;
            case KeyCode.A: CycleLidarView(); break;
            case KeyCode.S: CycleDots(); break;
            case KeyCode.D: CycleBeams(); break;
            case KeyCode.F: MoveRing(-1); break;
            case KeyCode.G: MoveRing(1); break;
            case KeyCode.Z: ToggleDashboard(); break;
            case KeyCode.X: ToggleMinimap(); break;
            case KeyCode.C: ToggleTrackOverlay(); break;
            case KeyCode.V: ToggleRoute(); break;
            case KeyCode.B: ToggleBoxes(); break;
            case KeyCode.Tab: if (e.shift) ToggleEffects(); else ToggleHud(); break;
            default: return;
        }
        e.Use();
    }

    static string BeamName(BeamMode m)
    {
        switch (m)
        {
            case BeamMode.Ring: return "Row";
            case BeamMode.Sparse: return "Grid";
            case BeamMode.Objects: return "Objects";
            default: return "OFF";
        }
    }

    static string OnOff(bool on) { return on ? "ON" : "OFF"; }

    // first letter big, the rest small ("Front_camera")
    static string Sentence(string s) { return string.IsNullOrEmpty(s) ? s : char.ToUpperInvariant(s[0]) + s.Substring(1).ToLowerInvariant(); }

    const float PanelW = 310f, DividerH = 11f;

    // one row of the view panel: label, the button with the current value (centred), the key
    static bool ViewRow(ref Rect inner, string label, string value, string key)
    {
        Rect[] c = UiTheme.Cols(UiTheme.NextRow(ref inner), 92f, 0f, 44f);
        GUI.Label(c[0], label, UiTheme.Muted);
        bool clicked = UiTheme.Press(c[1], value);
        GUI.Label(c[2], key, UiTheme.Right);
        return clicked;
    }

    // a little space, a thin line, a little space (the row before has its usual gap already)
    static void Divider(ref Rect inner)
    {
        UiTheme.Separator(new Rect(inner.x, inner.y + 2f, inner.width, 1f));
        inner.y += DividerH;
        inner.height -= DividerH;
    }

    static GUIStyle centredValue;

    void OnGUI()
    {
        GUI.depth = 10;      // behind the camera preview that the camera sensor draws, so tracks and panels never cover it
        UiTheme.Begin();
        DrawUi();
        UiTheme.End();
    }

    void DrawUi()
    {
        Event e = Event.current;
        if (hotkeys && e.type == EventType.KeyDown && GUIUtility.keyboardControl == 0 && !e.control && !e.alt) HandleKey(e);
        UiTheme.Ensure();

        if (!hudVisible) return;

        if (trackOverlayOn && !effectsHidden && cams.Count > 0) dashboard.DrawOverlay(cams[camIndex], vehicle, network);      // first: the pictures and panels are drawn over it

        // corner pictures: the camera preview (drawn by the camera sensor) is on top, the lidar view goes below it
        float top = 10f;
        if (cameraSensor != null && cameraSensor.showPreview && cameraSensor.LastFrame != null && cameraSensor.LastFrame.format == "jpeg")
            top += 320f * cameraSensor.LastFrame.height / cameraSensor.LastFrame.width + 28f;
        if (lidarView != LidarViewMode.Off && lidar != null && lidar.LastScan != null)
            panel.Draw(UiTheme.W - 10f, top, lidarView, lidar.LastScan.frameId, birdsEyeRangeM);

        if (dashboardOn) dashboard.Draw(new Rect(12f, 12f, Dashboard.Width, Dashboard.Height), vehicle, network);
        if (minimapOn) dashboard.DrawMap(new Rect(12f, 12f + (dashboardOn ? Dashboard.Height + 8f : 0f), Dashboard.Width, Dashboard.MapHeight), vehicle, network);

        if (centredValue == null) centredValue = new GUIStyle(UiTheme.Value) { alignment = TextAnchor.MiddleCenter };

        // the panel with the switches: a fixed width, exactly as high as its rows and dividers
        float h = UiTheme.PanelHeight(11) + 3f * DividerH;
        Rect inner = UiTheme.Panel(new Rect(UiTheme.W - PanelW - 12f, UiTheme.H - h - 12f, PanelW, h));

        string camName = cams.Count > 0 ? Sentence(cams[camIndex].name) + " (" + (camIndex + 1) + "/" + cams.Count + ")" : "None found";
        if (ViewRow(ref inner, "Camera view", OnOff(cameraView), "Q")) ToggleCameraView();
        if (ViewRow(ref inner, "Cycle camera", camName, "W  1-5")) NextCamera(Event.current.shift ? -1 : 1);
        Divider(ref inner);

        if (ViewRow(ref inner, "Lidar view", LidarViewName(), "A")) CycleLidarView();
        if (ViewRow(ref inner, "Lidar dots", dots == DotMode.Off ? "OFF" : dots == DotMode.Objects ? "Objects" : "All returns", "S")) CycleDots();
        if (ViewRow(ref inner, "Lidar beams", BeamName(beams) + (beams != BeamMode.Off ? "  (" + lastBeamCount + ")" : ""), "D")) CycleBeams();

        // the beam row: always there, dimmed while the beams are not on "row"
        Color old = GUI.color;
        if (beams != BeamMode.Ring) GUI.color = new Color(1f, 1f, 1f, 0.4f);
        Rect[] c = UiTheme.Cols(UiTheme.NextRow(ref inner), 92f, 0f, 44f);
        GUI.Label(c[0], "Beam row", UiTheme.Muted);
        Rect[] b = UiTheme.Cols(c[1], 30f, 0f, 30f);
        if (UiTheme.Press(b[0], "<")) MoveRing(-1);
        GUI.Label(b[1], Rows > 0 ? (RingRow() + 1) + " of " + Rows : "-", centredValue);
        if (UiTheme.Press(b[2], ">")) MoveRing(1);
        GUI.Label(c[2], "F  G", UiTheme.Right);
        GUI.color = old;
        Divider(ref inner);

        if (ViewRow(ref inner, "Dashboard", OnOff(dashboardOn), "Z")) ToggleDashboard();
        if (ViewRow(ref inner, "Minimap", OnOff(minimapOn), "X")) ToggleMinimap();
        Divider(ref inner);

        if (ViewRow(ref inner, "Track", OnOff(trackOverlayOn), "C")) ToggleTrackOverlay();
        if (ViewRow(ref inner, "Route", OnOff(routeLine), "V")) ToggleRoute();
        if (ViewRow(ref inner, "Hazard boxes", OnOff(groundTruthBoxes), "B")) ToggleBoxes();
    }

    static string LidarViewName()
    {
        switch (lidarView)
        {
            case LidarViewMode.RangeImage: return "Range image";
            case LidarViewMode.BirdsEye: return "Bird's-eye";
            case LidarViewMode.Both: return "Both";
            default: return "OFF";
        }
    }
}