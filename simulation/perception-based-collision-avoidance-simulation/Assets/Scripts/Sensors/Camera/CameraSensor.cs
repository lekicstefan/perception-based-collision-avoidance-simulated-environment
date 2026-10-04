using System.Collections.Concurrent;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

[DefaultExecutionOrder(25)] // after Vehicle and the scripted objects of the same step
public class CameraSensor : MonoBehaviour
{
    public enum ReadbackOrder { Auto, TopDown, BottomUp }

    public SensorRig rig;
    public LayerMask excludeLayers;                       // the ego car (Ego) is not rendered
    public Color fogColor = new Color(0.72f, 0.75f, 0.78f);
    public ReadbackOrder readbackOrder = ReadbackOrder.Auto;
    public bool showPreview = true;                       // decodes the JPEG that is sent, top right
    public int previewEveryN = 3;
    public int logEveryFrames = 60;

    public event System.Action<CameraFrame> FrameReady;   // raised on the main thread
    public CameraFrame LastFrame { get; private set; }
    public PoseEstimator poseEstimator;

    struct Job { public int frameId; public double timestamp; public byte[] rgba; public long wallStart; public PoseState pose; }

    CameraConfig cfg;
    Camera cam;
    RenderTexture rt;
    SensorSchedule schedule;
    System.Threading.Thread worker;
    BlockingCollection<Job> jobs;
    readonly ConcurrentQueue<CameraFrame> done = new ConcurrentQueue<CameraFrame>();
    readonly ConcurrentQueue<byte[]> pool = new ConcurrentQueue<byte[]>();
    Texture2D previewTex;
    int frameId, dropped, readbackErrors, delivered;
    int camSeed;
    bool readbackTopDown, encoderFirstRowIsBottom;
    float latSum, latMin = 1e9f, latMax;
    volatile float lastNoiseRms;
    byte[] rgbBuf, flipBuf;
    float[] blurTmp;

    void Start()
    {
        cfg = rig.Config.camera;
        if (!cfg.enabled) return;
        if (!SystemInfo.supportsAsyncGPUReadback)
            UnityEngine.Debug.LogWarning("CameraSensor: this graphics API does not support AsyncGPUReadback");

        var go = new GameObject("SensorCamera");
        go.transform.SetParent(rig.CameraMount, false);
        cam = go.AddComponent<Camera>();
        cam.enabled = false;                         // rendered by hand, at the sensor rate only
        cam.fieldOfView = cfg.VerticalFovDeg;        // Unity's field of view is the vertical one
        cam.aspect = (float)cfg.width / cfg.height;
        cam.nearClipPlane = 0.3f;
        cam.farClipPlane = 250f;
        cam.allowHDR = false;
        cam.allowMSAA = false;
        cam.cullingMask = ~excludeLayers.value;

        rt = new RenderTexture(cfg.width, cfg.height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt.antiAliasing = 1;
        rt.Create();
        cam.targetTexture = rt;

        schedule = new SensorSchedule(cfg.rateHz);
        camSeed = rig.DerivedSeed("camera");

        readbackTopDown = readbackOrder == ReadbackOrder.Auto ? DetectReadbackTopDown()
                                                      : readbackOrder == ReadbackOrder.TopDown;
        encoderFirstRowIsBottom = DetectEncoderFirstRowIsBottom();
        UnityEngine.Debug.Log("CAMERA " + cfg.width + "x" + cfg.height + " " + cfg.format + " | readback rows top-down: " + readbackTopDown +
            " | JPEG encoder treats the first row as the " + (encoderFirstRowIsBottom ? "bottom" : "top") + " row");

        previewTex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        jobs = new BlockingCollection<Job>(4);
        worker = new System.Threading.Thread(WorkerLoop) { IsBackground = true, Name = "CameraEncoder" };
        worker.Start();
    }

    // Encodes a synthetic image (first rows red, last rows blue) and decodes it again to find out how the encoder
    // orders rows. Main thread only.
    public static bool DetectEncoderFirstRowIsBottom()
    {
        const int w = 16, h = 16;
        var rgb = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = (y * w + x) * 3;
                bool first = y < h / 2;
                rgb[i] = (byte)(first ? 255 : 0);
                rgb[i + 2] = (byte)(first ? 0 : 255);
            }
        byte[] jpg = ImageConversion.EncodeArrayToJPG(rgb, GraphicsFormat.R8G8B8_SRGB, w, h, 0, 100);
        var tex = new Texture2D(2, 2);
        tex.LoadImage(jpg);
        Color top = tex.GetPixel(w / 2, h - 2);      // y = h-1 is the top row of a decoded texture
        UnityEngine.Object.Destroy(tex);
        return !(top.r > top.b);                     // red on top: first memory row is the top row
    }

    // Draws a test image (top half red, bottom half blue) into a render texture, reads it back and checks which
    // row comes first in memory. Main thread only.
    public static bool DetectReadbackTopDown()
    {
        const int s = 16;
        var src = new Texture2D(s, s, TextureFormat.RGBA32, false);
        var px = new Color32[s * s];
        for (int y = 0; y < s; y++)          // y = 0 is the bottom row of a Texture2D
            for (int x = 0; x < s; x++)
                px[y * s + x] = y >= s / 2 ? new Color32(255, 0, 0, 255) : new Color32(0, 0, 255, 255);
        src.SetPixels32(px);
        src.Apply();

        RenderTexture tmp = RenderTexture.GetTemporary(s, s, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        Graphics.Blit(src, tmp);
        AsyncGPUReadbackRequest req = AsyncGPUReadback.Request(tmp, 0, TextureFormat.RGBA32);
        req.WaitForCompletion();

        bool topDown = false;
        if (!req.hasError)
        {
            var data = req.GetData<byte>();
            topDown = data[0] > data[2];     // first pixel of the first row is red: the top row comes first
        }
        RenderTexture.ReleaseTemporary(tmp);
        UnityEngine.Object.Destroy(src);
        return topDown;
    }

    void ApplyFog()
    {
        float beta = SimEnvironment.FogExtinction;
        bool fog = beta > 0f;
        RenderSettings.fog = fog;
        if (fog)
        {
            RenderSettings.fogMode = FogMode.Exponential;
            RenderSettings.fogDensity = beta;
            RenderSettings.fogColor = fogColor;
        }
        cam.clearFlags = fog ? CameraClearFlags.SolidColor : CameraClearFlags.Skybox;
        cam.backgroundColor = fogColor;
    }

    void FixedUpdate()
    {
        if (cam == null) return;
        double now = Time.fixedTimeAsDouble;
        if (!schedule.Due(now, Time.fixedDeltaTime)) return;

        ApplyFog();
        cam.Render();
        int id = frameId++;
        long wall = System.Diagnostics.Stopwatch.GetTimestamp();
        PoseState p = poseEstimator != null ? poseEstimator.Current : default(PoseState);
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => OnReadback(req, id, now, wall, p));
    }

    void OnReadback(AsyncGPUReadbackRequest req, int id, double t, long wall, PoseState p)
    {
        if (jobs == null || jobs.IsAddingCompleted) return;
        if (req.hasError) { readbackErrors++; return; }
        byte[] buf;
        if (!pool.TryDequeue(out buf)) buf = new byte[cfg.width * cfg.height * 4];
        req.GetData<byte>().CopyTo(buf);
        if (!jobs.TryAdd(new Job { frameId = id, timestamp = t, rgba = buf, wallStart = wall, pose = p }))
        {
            dropped++;
            pool.Enqueue(buf);
        }
    }

    // ---- worker thread ----
    void WorkerLoop()
    {
        foreach (Job job in jobs.GetConsumingEnumerable())
        {
            try { done.Enqueue(Process(job)); }
            catch (System.Exception e) { UnityEngine.Debug.LogError("Camera worker: " + e); }
        }
    }

    CameraFrame Process(Job job)
    {
        int w = cfg.width, h = cfg.height, n = w * h * 3;
        bool raw = cfg.format == "raw";
        byte[] rgb = raw ? new byte[n] : (rgbBuf ?? (rgbBuf = new byte[n]));

        // RGBA -> RGB, rows top to bottom
        byte[] src = job.rgba;
        for (int y = 0; y < h; y++)
        {
            int si = (readbackTopDown ? y : h - 1 - y) * w * 4, di = y * w * 3;
            for (int x = 0; x < w; x++) { rgb[di] = src[si]; rgb[di + 1] = src[si + 1]; rgb[di + 2] = src[si + 2]; si += 4; di += 3; }
        }
        if (pool.Count < 8) pool.Enqueue(src);

        PostProcessingConfig pp = cfg.postProcessing;
        if (pp.exposureEnabled) CameraPostProcessing.Exposure(rgb, pp.exposureGain);
        if (pp.blurEnabled && pp.blurSigmaPx > 0f) CameraPostProcessing.Blur(rgb, w, h, pp.blurSigmaPx, ref blurTmp);
        if (pp.noiseEnabled && pp.noiseStd > 0f)
            lastNoiseRms = CameraPostProcessing.Noise(rgb, pp.noiseStd, (uint)SeedUtil.Derive(camSeed, "f" + job.frameId));

        byte[] data;
        if (raw) data = rgb;
        else
        {
            byte[] enc = rgb;
            if (encoderFirstRowIsBottom)
            {
                if (flipBuf == null) flipBuf = new byte[n];
                CameraPostProcessing.FlipRows(rgb, flipBuf, w, h);
                enc = flipBuf;
            }
            data = ImageConversion.EncodeArrayToJPG(enc, GraphicsFormat.R8G8B8_SRGB, (uint)w, (uint)h, 0, cfg.jpegQuality);
        }
        float ms = (float)((System.Diagnostics.Stopwatch.GetTimestamp() - job.wallStart) * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
        return new CameraFrame { timestamp = job.timestamp, frameId = job.frameId, width = w, height = h, format = cfg.format, data = data, latencyMs = ms, pose = job.pose };
    }

    // ---- main thread ----
    void Update()
    {
        CameraFrame f;
        while (done.TryDequeue(out f))
        {
            LastFrame = f;
            delivered++;
            latSum += f.latencyMs;
            latMin = Mathf.Min(latMin, f.latencyMs);
            latMax = Mathf.Max(latMax, f.latencyMs);
            if (showPreview && f.format == "jpeg" && f.frameId % Mathf.Max(1, previewEveryN) == 0) previewTex.LoadImage(f.data);

            if (f.frameId < 3 || (logEveryFrames > 0 && f.frameId % logEveryFrames == 0))
            {
                uint h = 2166136261u;
                unchecked { foreach (byte b in f.data) { h ^= b; h *= 16777619u; } }
                UnityEngine.Debug.Log("CAMERA frame " + f.frameId + " t=" + f.timestamp.ToString("F3") + " | " + f.format + " " + f.data.Length +
                    " bytes | checksum " + h.ToString("X8") + " | latency ms min/avg/max " + latMin.ToString("F1") + "/" +
                    (latSum / Mathf.Max(1, delivered)).ToString("F1") + "/" + latMax.ToString("F1") +
                    " | dropped " + dropped + " | readback errors " + readbackErrors +
                    (cfg.postProcessing.noiseEnabled ? " | noise rms " + lastNoiseRms.ToString("F2") : "") +
                    " | pose stamp " + f.pose.timestamp.ToString("F3"));
                latSum = 0f; latMin = 1e9f; latMax = 0f; delivered = 0;
            }
            if (FrameReady != null) FrameReady(f);
        }
    }

    void OnGUI()
    {
        if (!showPreview || previewTex == null || LastFrame == null || LastFrame.format != "jpeg") return;
        float w = 320f, h = w * cfg.height / cfg.width;
        var r = new Rect(Screen.width - w - 10f, 10f, w, h);
        GUI.DrawTexture(r, previewTex);
        GUI.Label(new Rect(r.x, r.yMax + 2f, w, 20f), "camera frame " + LastFrame.frameId + " | " + LastFrame.data.Length + " bytes");
    }

    [ContextMenu("Save sample frame")]
    void SaveSample()
    {
        if (LastFrame == null || LastFrame.format != "jpeg") { UnityEngine.Debug.LogWarning("No JPEG frame yet"); return; }
        string dir = System.IO.Path.Combine(System.IO.Directory.GetParent(ConfigPaths.ConfigDir()).FullName, "testing", "camera testing");
        System.IO.Directory.CreateDirectory(dir);
        string path = System.IO.Path.Combine(dir, "camera_sample.jpg");
        System.IO.File.WriteAllBytes(path, LastFrame.data);
        UnityEngine.Debug.Log("Saved " + path);
    }

    void OnDestroy()
    {
        if (jobs != null) jobs.CompleteAdding();
        if (worker != null) worker.Join(500);
        if (rt != null) rt.Release();
    }
}