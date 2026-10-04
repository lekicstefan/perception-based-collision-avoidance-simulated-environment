using System.Text;
using UnityEngine;

public class CameraPostProcessingTest : MonoBehaviour
{
    [ContextMenu("Run post-processing self-test")]
    void Run()
    {
        var sb = new StringBuilder("POST-PROCESSING SELF-TEST\n");

        // exposure
        var lutImg = new byte[] { 32, 32, 32, 64, 64, 64, 128, 128, 128, 192, 192, 192, 255, 255, 255 };
        CameraPostProcessing.Exposure(lutImg, 0.25f);
        sb.AppendLine("exposure gain 0.25: 32->" + lutImg[0] + ", 64->" + lutImg[3] + ", 128->" + lutImg[6] + ", 192->" + lutImg[9] + ", 255->" + lutImg[12] +
                      "  (expected 12, 30, 66, 102, 137)");

        // blur of a step edge between x = 31 and x = 32
        int w = 64, h = 8;
        var edge = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
            for (int x = 32; x < w; x++)
                for (int c = 0; c < 3; c++) edge[(y * w + x) * 3 + c] = 255;
        float[] tmp = null;
        CameraPostProcessing.Blur(edge, w, h, 1.5f, ref tmp);
        var line = new StringBuilder();
        for (int x = 28; x <= 35; x++) line.Append(edge[(4 * w + x) * 3]).Append(' ');
        sb.AppendLine("blur sigma 1.5, step edge x=28..35: " + line + " (expected 2 11 39 94 161 216 244 253, +-1)");

        // noise on a flat image
        foreach (float std in new[] { 3f, 8f })
        {
            var flat = new byte[64 * 64 * 3];
            for (int i = 0; i < flat.Length; i++) flat[i] = 128;
            float applied = CameraPostProcessing.Noise(flat, std, 12345);
            double mean = 0, sq = 0;
            foreach (byte b in flat) mean += b;
            mean /= flat.Length;
            foreach (byte b in flat) sq += (b - mean) * (b - mean);
            sb.AppendLine("noise std " + std + " seed 12345: mean " + mean.ToString("F2") + ", rms " + System.Math.Sqrt(sq / flat.Length).ToString("F2") +
                          " (reported " + applied.ToString("F2") + ")  expected mean ~128.0, rms " + (std == 3f ? "3.03" : "8.05"));
        }

        // flip
        var src = new byte[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 };   // 2 x 2, rows (1,2) and (3,4)
        var dst = new byte[12];
        CameraPostProcessing.FlipRows(src, dst, 2, 2);
        sb.AppendLine("row flip: " + (dst[0] == 3 && dst[3] == 4 && dst[6] == 1 && dst[9] == 2 ? "PASS" : "FAIL"));

        sb.AppendLine("JPEG encoder treats the first memory row as the " + (CameraSensor.DetectEncoderFirstRowIsBottom() ? "bottom" : "top") +
                      " row; graphicsUVStartsAtTop = " + SystemInfo.graphicsUVStartsAtTop);
        sb.AppendLine("readback rows top-down (measured): " + CameraSensor.DetectReadbackTopDown());
        UnityEngine.Debug.Log(sb.ToString());
    }
}