using UnityEngine;

// The lidar picture in the corner of the screen (debug view, never sent anywhere): either the range image (what the processor receives,
// 32 rows x 600 columns) or a bird's-eye plot of the returns around the car with the heading up.
public class LidarPanel
{
    const int BevSize = 256;
    const int RangeRowScale = 3;     // screen pixels per range-image row

    Texture2D rangeTex, bevTex;
    Color32[] rangeBuf, bevBuf, bevBg;
    int rangeFrame = -1, bevFrame = -1;
    float bevRange = -1f;
    int validCount;

    static readonly Color32 GroundDot = new Color32(60, 190, 90, 255);
    static readonly Color32 ObjectDot = new Color32(255, 90, 40, 255);

    static Color32 RangeColor(ushort v, float maxRange)
    {
        if (v == LidarScan.MaxRangeOrSky) return new Color32(8, 8, 18, 255);        // nothing within range
        if (v == LidarScan.NoReturn) return new Color32(120, 25, 120, 255);        // dropout, fog or grazing angle: not free space
        float d = v * 0.01f;
        float t = Mathf.Clamp01(Mathf.Log(1f + d) / Mathf.Log(1f + maxRange));
        return (Color32)Color.HSVToRGB(0.66f * t, 0.9f, 1f);                        // near = red, far = blue
    }

    public void Refresh(LidarSensor lidar, LidarOverlay cache, ViewControls.LidarViewMode mode, Transform ego, float birdsEyeRangeM)
    {
        LidarScan scan = lidar.LastScan;
        if (scan == null) return;
        if (mode == ViewControls.LidarViewMode.RangeImage && scan.frameId != rangeFrame) BuildRange(lidar, scan);
        else if (mode == ViewControls.LidarViewMode.BirdsEye && (scan.frameId != bevFrame || !Mathf.Approximately(birdsEyeRangeM, bevRange))) BuildBirdsEye(cache, ego, birdsEyeRangeM, scan.frameId);
    }

    void BuildRange(LidarSensor lidar, LidarScan scan)
    {
        rangeFrame = scan.frameId;
        int rows = scan.rows, cols = scan.cols;
        if (rangeTex == null || rangeTex.width != cols || rangeTex.height != rows)
        {
            rangeTex = new Texture2D(cols, rows, TextureFormat.RGBA32, false);
            rangeTex.filterMode = FilterMode.Point;
            rangeBuf = new Color32[cols * rows];
        }
        float maxRange = lidar.rig.Config.lidar.maxRangeM;
        validCount = 0;
        for (int r = 0; r < rows; r++)
        {
            int dst = (rows - 1 - r) * cols;      // row 0 is the highest beam and goes on top; texture row 0 is the bottom
            for (int c = 0; c < cols; c++)
            {
                ushort v = scan.ranges[r * cols + c];
                if (v != LidarScan.NoReturn && v != LidarScan.MaxRangeOrSky) validCount++;
                rangeBuf[dst + c] = RangeColor(v, maxRange);
            }
        }
        rangeTex.SetPixels32(rangeBuf);
        rangeTex.Apply(false);
    }

    void BuildBirdsEye(LidarOverlay cache, Transform ego, float range, int frameId)
    {
        bevFrame = frameId;
        if (bevTex == null)
        {
            bevTex = new Texture2D(BevSize, BevSize, TextureFormat.RGBA32, false);
            bevTex.filterMode = FilterMode.Point;
            bevBuf = new Color32[BevSize * BevSize];
        }
        if (bevBg == null || !Mathf.Approximately(range, bevRange)) BuildBackground(range);
        bevRange = range;
        System.Array.Copy(bevBg, bevBuf, bevBuf.Length);

        Vector3 fwd = ego.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
        fwd.Normalize();
        Vector3 right = Vector3.Cross(Vector3.up, fwd);
        float pxPerM = BevSize / (2f * range);
        validCount = cache.Count;

        for (int pass = 1; pass <= 2; pass++)           // ground first, objects on top
        {
            Color32 color = pass == 1 ? GroundDot : ObjectDot;
            for (int k = 0; k < cache.Count; k++)
            {
                if (cache.Class[k] != pass) continue;
                Vector3 rel = cache.Points[k] - cache.Origin;
                int px = (int)(BevSize * 0.5f + Vector3.Dot(rel, right) * pxPerM);
                int py = (int)(BevSize * 0.5f + Vector3.Dot(rel, fwd) * pxPerM);
                Plot(px, py, color);
                Plot(px + 1, py, color);
                Plot(px, py + 1, color);
                Plot(px + 1, py + 1, color);
            }
        }
        bevTex.SetPixels32(bevBuf);
        bevTex.Apply(false);
    }

    void Plot(int x, int y, Color32 c)
    {
        if (x < 0 || y < 0 || x >= BevSize || y >= BevSize) return;
        bevBuf[y * BevSize + x] = c;
    }

    // dark background, range circles every 10 m, the axes and a marker for the car
    void BuildBackground(float range)
    {
        bevBg = new Color32[BevSize * BevSize];
        var bg = new Color32(22, 24, 30, 255);
        for (int i = 0; i < bevBg.Length; i++) bevBg[i] = bg;
        float pxPerM = BevSize / (2f * range);
        float centre = BevSize * 0.5f;
        var grid = new Color32(60, 64, 74, 255);
        for (float ringM = 10f; ringM < range + 0.01f; ringM += 10f)
        {
            for (int a = 0; a < 720; a++)
            {
                float ang = a * Mathf.PI / 360f;
                int x = (int)(centre + Mathf.Cos(ang) * ringM * pxPerM), y = (int)(centre + Mathf.Sin(ang) * ringM * pxPerM);
                if (x >= 0 && y >= 0 && x < BevSize && y < BevSize) bevBg[y * BevSize + x] = grid;
            }
        }
        for (int i = 0; i < BevSize; i++)
        {
            bevBg[(BevSize / 2) * BevSize + i] = grid;
            bevBg[i * BevSize + BevSize / 2] = grid;
        }
        var car = new Color32(240, 240, 240, 255);
        for (int dy = -5; dy <= 5; dy++)
            for (int dx = -2; dx <= 2; dx++)
                bevBg[(BevSize / 2 + dy) * BevSize + (BevSize / 2 + dx)] = car;
    }

    // Draws the panel with its right edge at rightX and its top at top. Returns the height it used.
    public float Draw(float rightX, float top, ViewControls.LidarViewMode mode, int frameId, float birdsEyeRangeM)
    {
        if (mode == ViewControls.LidarViewMode.RangeImage && rangeTex != null)
        {
            float w = Mathf.Min(rangeTex.width, Screen.width - 20f);
            float h = rangeTex.height * RangeRowScale;
            var r = new Rect(rightX - w, top, w, h);
            GUI.DrawTexture(r, rangeTex, ScaleMode.StretchToFill);
            GUI.Label(new Rect(r.x, r.yMax + 1f, w, 22f), "lidar range image, scan " + frameId + " | " + validCount + " returns | red near, blue far, black nothing, purple no return");
            return h + 24f;
        }
        if (mode == ViewControls.LidarViewMode.BirdsEye && bevTex != null)
        {
            var r = new Rect(rightX - BevSize, top, BevSize, BevSize);
            GUI.DrawTexture(r, bevTex, ScaleMode.StretchToFill);
            GUI.Label(new Rect(r.x - 90f, r.yMax + 1f, BevSize + 90f, 22f), "bird's-eye, " + birdsEyeRangeM.ToString("0") + " m radius, heading up | scan " + frameId);
            return BevSize + 24f;
        }
        return 0f;
    }
}