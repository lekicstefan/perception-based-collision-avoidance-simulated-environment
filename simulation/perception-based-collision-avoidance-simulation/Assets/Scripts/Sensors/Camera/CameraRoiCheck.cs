using UnityEngine;

public class CameraRoiCheck : MonoBehaviour
{
    public CameraSensor cameraSensor;
    public int roiSize = 48;
    public int roiCentreX = 320;     // pixels from the left
    public int roiCentreY = 120;     // pixels from the top (above the horizon, inside the wall)
    public int everyNFrames = 5;
    public int reportEvery = 12;     // evaluated frames per report

    Texture2D tex;
    readonly double[] sumMean = new double[3];
    readonly double[] sumStd = new double[3];
    int n;

    void OnEnable() { if (cameraSensor != null) cameraSensor.FrameReady += OnFrame; }
    void OnDisable() { if (cameraSensor != null) cameraSensor.FrameReady -= OnFrame; }

    void OnFrame(CameraFrame f)
    {
        if (f.format != "jpeg" || f.frameId % Mathf.Max(1, everyNFrames) != 0) return;
        if (tex == null) tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        tex.LoadImage(f.data);

        int x0 = Mathf.Clamp(roiCentreX - roiSize / 2, 0, tex.width - roiSize);
        int yTop = Mathf.Clamp(roiCentreY - roiSize / 2, 0, tex.height - roiSize);
        Color[] px = tex.GetPixels(x0, tex.height - yTop - roiSize, roiSize, roiSize);   // textures count rows from the bottom

        for (int c = 0; c < 3; c++)
        {
            double s = 0, s2 = 0;
            for (int i = 0; i < px.Length; i++)
            {
                double v = 255.0 * (c == 0 ? px[i].r : (c == 1 ? px[i].g : px[i].b));
                s += v;
                s2 += v * v;
            }
            double mean = s / px.Length;
            sumMean[c] += mean;
            sumStd[c] += System.Math.Sqrt(System.Math.Max(0.0, s2 / px.Length - mean * mean));
        }
        n++;
        if (n % Mathf.Max(1, reportEvery) == 0)
        {
            UnityEngine.Debug.Log("CAMERA ROI (" + roiSize + "x" + roiSize + " at " + roiCentreX + "," + roiCentreY + ", " + n + " frames) mean R/G/B " +
                (sumMean[0] / n).ToString("F1") + " / " + (sumMean[1] / n).ToString("F1") + " / " + (sumMean[2] / n).ToString("F1") +
                " | std R/G/B " + (sumStd[0] / n).ToString("F2") + " / " + (sumStd[1] / n).ToString("F2") + " / " + (sumStd[2] / n).ToString("F2"));
        }
    }
}