using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

// What you see while a run is going: which camera, and which debug overlays. Everything here is for the person watching. None of it is
// sent to the processor and none of it is drawn into the sensor camera (which renders to a texture), so a run looks the same to the system
// whatever is switched on.
//
// Keys (click the Game view first; ignored while a text field has the keyboard):
//   1-9 or C / Shift+C   camera (the cameras below the Vehicle, in name order)
//   D   lidar dots: off / objects / all returns
//   B   lidar beams: off / one ring / sparse grid / objects      , and .   move the ring to another beam row
//   L   lidar view in the corner: off / range image / bird's-eye
//   V   camera view in the corner (the image the processor receives)
//   G   boxes around the Hazard objects (ground truth)
//   T   the route
//   H   hide / show the whole HUD (for screenshots)
// The choices are kept when the scene is restarted. Run-control keys (R, P, X, +, -, 0) belong to the ScenarioManager.
[DefaultExecutionOrder(-250)]
public class ViewControls : MonoBehaviour
{
    public enum DotMode { Off, Objects, All }
    public enum BeamMode { Off, Ring, Sparse, Objects }
    public enum LidarViewMode { Off, RangeImage, BirdsEye }

    [Header("Wiring (found in the scene when left empty)")]
    public Vehicle vehicle;
    public RouteDriver driver;
    public LidarSensor lidar;
    public CameraSensor cameraSensor;
    [Tooltip("Parent of the view cameras. Empty: every camera below the Vehicle that renders to the screen.")]
    public Transform cameraRoot;

    [Header("Start values (first run of a session; restarts keep what you chose)")]
    public int startCamera = 0;
    public DotMode startDots = DotMode.Objects;
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
    static bool hudVisible = true;
    static int ringRow = -1;

    public static bool HudVisible { get { return hudVisible; } }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics() { initialised = false; hudVisible = true; ringRow = -1; }

    readonly List<Camera> cams = new List<Camera>();
    readonly LidarOverlay overlay = new LidarOverlay();
    readonly LidarPanel panel = new LidarPanel();
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
            dots = startDots;
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

        FindCameras();
        if (lidar != null) statsDefault = lidar.showStats;
    }

    void FindCameras()
    {
        cams.Clear();
        Transform root = cameraRoot != null ? cameraRoot : (vehicle != null ? vehicle.transform : null);
        Camera[] found = root != null ? root.GetComponentsInChildren<Camera>(true) : FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        foreach (Camera c in found)
            if (c.targetTexture == null && c.name != "SensorCamera") cams.Add(c);     // the sensor camera renders to a texture and is not a view
        cams.Sort((a, b) => string.CompareOrdinal(a.name, b.name));

        if (cams.Count == 0)
        {
            UnityEngine.Debug.LogWarning("VIEW CONTROLS: no camera found below the Vehicle (or the Camera Root), the camera keys do nothing");
            return;
        }
        camIndex = Mathf.Clamp(camIndex, 0, cams.Count - 1);
        ApplyCamera();
    }

    // Only the chosen camera renders, and only its audio listener is on (several of them produce a warning every frame).
    void ApplyCamera()
    {
        for (int i = 0; i < cams.Count; i++)
        {
            bool on = i == camIndex;
            cams[i].enabled = on;
            AudioListener al = cams[i].GetComponent<AudioListener>();
            if (al != null) al.enabled = on;
        }
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
    void CycleLidarView() { lidarView = (LidarViewMode)(((int)lidarView + 1) % 3); }
    void ToggleCameraView() { cameraView = !cameraView; }
    void ToggleBoxes() { groundTruthBoxes = !groundTruthBoxes; }
    void ToggleRoute() { routeLine = !routeLine; }
    void ToggleHud() { hudVisible = !hudVisible; }

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
            bool need = dots != DotMode.Off || beams != BeamMode.Off || lidarView == LidarViewMode.BirdsEye;
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
        }
        else if (k == KeyCode.C) { NextCamera(e.shift ? -1 : 1); e.Use(); }
        else if (k == KeyCode.D) { CycleDots(); e.Use(); }
        else if (k == KeyCode.B) { CycleBeams(); e.Use(); }
        else if (k == KeyCode.L) { CycleLidarView(); e.Use(); }
        else if (k == KeyCode.V) { ToggleCameraView(); e.Use(); }
        else if (k == KeyCode.G) { ToggleBoxes(); e.Use(); }
        else if (k == KeyCode.T) { ToggleRoute(); e.Use(); }
        else if (k == KeyCode.H) { ToggleHud(); e.Use(); }
        else if (k == KeyCode.None)
        {
            if (e.character == ',') { MoveRing(-1); e.Use(); }
            else if (e.character == '.') { MoveRing(1); e.Use(); }
        }
    }

    static bool Row(string label, string value)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(label, GUILayout.Width(92f));
        bool clicked = GUILayout.Button(value);
        GUILayout.EndHorizontal();
        return clicked;
    }

    static string BeamName(BeamMode m)
    {
        switch (m)
        {
            case BeamMode.Ring: return "one ring";
            case BeamMode.Sparse: return "sparse grid";
            case BeamMode.Objects: return "objects only";
            default: return "off";
        }
    }

    void OnGUI()
    {
        Event e = Event.current;
        if (hotkeys && e.type == EventType.KeyDown && GUIUtility.keyboardControl == 0 && !e.control && !e.alt) HandleKey(e);

        if (!hudVisible)
        {
            if (e.type == EventType.Repaint) GUI.Label(new Rect(Screen.width - 130f, Screen.height - 24f, 120f, 20f), "H = show HUD");
            return;
        }

        // corner pictures: the camera preview (drawn by the camera sensor) is on top, the lidar view goes below it
        float top = 10f;
        if (cameraSensor != null && cameraSensor.showPreview && cameraSensor.LastFrame != null && cameraSensor.LastFrame.format == "jpeg")
            top += 320f * cameraSensor.LastFrame.height / cameraSensor.LastFrame.width + 28f;
        if (lidarView != LidarViewMode.Off && lidar != null && lidar.LastScan != null)
            panel.Draw(Screen.width - 10f, top, lidarView, lidar.LastScan.frameId, birdsEyeRangeM);

        // the panel with the switches
        float w = 360f, h = lidarView != LidarViewMode.Off || beams == BeamMode.Ring ? 250f : 226f;
        GUILayout.BeginArea(new Rect(Screen.width - w - 10f, Screen.height - h - 10f, w, h), GUI.skin.box);
        GUILayout.Label("VIEW   (click the Game view first, H hides everything)");

        string camName = cams.Count > 0 ? cams[camIndex].name + "  (" + (camIndex + 1) + "/" + cams.Count + ")" : "none found";
        if (Row("Camera [1-9, C]", camName)) NextCamera(Event.current.shift ? -1 : 1);
        if (Row("Lidar dots [D]", dots == DotMode.Off ? "off" : dots == DotMode.Objects ? "objects" : "all returns")) CycleDots();
        if (Row("Lidar beams [B]", BeamName(beams) + (beams != BeamMode.Off ? "  (" + lastBeamCount + " drawn)" : ""))) CycleBeams();
        if (beams == BeamMode.Ring)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label("Ring beam row", GUILayout.Width(92f));
            if (GUILayout.Button("<", GUILayout.Width(30f))) MoveRing(-1);
            GUILayout.Label((RingRow() + 1) + " of " + Rows + "   [ , . ]");
            if (GUILayout.Button(">", GUILayout.Width(30f))) MoveRing(1);
            GUILayout.EndHorizontal();
        }
        if (Row("Lidar view [L]", lidarView == LidarViewMode.Off ? "off" : lidarView == LidarViewMode.RangeImage ? "range image" : "bird's-eye")) CycleLidarView();
        if (Row("Camera view [V]", cameraView ? "on" : "off")) ToggleCameraView();
        if (Row("Hazard boxes [G]", groundTruthBoxes ? "on" : "off")) ToggleBoxes();
        if (Row("Route [T]", routeLine ? "on" : "off")) ToggleRoute();
        GUILayout.EndArea();
    }
}