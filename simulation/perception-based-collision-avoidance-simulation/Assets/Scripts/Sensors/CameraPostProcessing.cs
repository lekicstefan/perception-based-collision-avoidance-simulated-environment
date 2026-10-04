using UnityEngine;

public static class CameraPostProcessing
{
    // gain applied in linear light, through a 256-entry lookup table
    public static void Exposure(byte[] img, float gain)
    {
        var lut = new byte[256];
        for (int v = 0; v < 256; v++)
        {
            float s = v / 255f;
            float lin = s <= 0.04045f ? s / 12.92f : Mathf.Pow((s + 0.055f) / 1.055f, 2.4f);
            lin = Mathf.Min(1f, lin * gain);
            float o = lin <= 0.0031308f ? lin * 12.92f : 1.055f * Mathf.Pow(lin, 1f / 2.4f) - 0.055f;
            lut[v] = (byte)Mathf.Clamp(Mathf.RoundToInt(o * 255f), 0, 255);
        }
        for (int i = 0; i < img.Length; i++) img[i] = lut[img[i]];
    }

    // separable Gaussian blur, kernel radius ceil(3 sigma), edge pixels repeated
    public static void Blur(byte[] img, int w, int h, float sigma, ref float[] tmp)
    {
        int r = Mathf.CeilToInt(3f * sigma);
        var k = new float[2 * r + 1];
        float sum = 0f;
        for (int i = -r; i <= r; i++) { k[i + r] = Mathf.Exp(-0.5f * i * i / (sigma * sigma)); sum += k[i + r]; }
        for (int i = 0; i < k.Length; i++) k[i] /= sum;
        if (tmp == null || tmp.Length != img.Length) tmp = new float[img.Length];
        int stride = w * 3;

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++)
                {
                    float acc = 0f;
                    for (int i = -r; i <= r; i++)
                    {
                        int xx = x + i;
                        if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
                        acc += k[i + r] * img[y * stride + xx * 3 + c];
                    }
                    tmp[y * stride + x * 3 + c] = acc;
                }

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++)
                {
                    float acc = 0f;
                    for (int i = -r; i <= r; i++)
                    {
                        int yy = y + i;
                        if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
                        acc += k[i + r] * tmp[yy * stride + x * 3 + c];
                    }
                    img[y * stride + x * 3 + c] = (byte)Mathf.Clamp(Mathf.RoundToInt(acc), 0, 255);
                }
    }

    // zero-mean approximately Gaussian noise (sum of four bytes of an xorshift32 stream), std in intensity levels.
    // Returns the RMS of the noise that was added (before clamping).
    public static float Noise(byte[] img, float std, uint seed)
    {
        uint s = seed == 0 ? 2463534242u : seed;
        float scale = std / 147.8f;   // std of the sum of four uniform bytes is 147.8
        double sq = 0.0;
        for (int i = 0; i < img.Length; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            int sumBytes = (int)(s & 0xFF) + (int)((s >> 8) & 0xFF) + (int)((s >> 16) & 0xFF) + (int)(s >> 24);
            int n = Mathf.RoundToInt((sumBytes - 510) * scale);
            sq += n * n;
            int v = img[i] + n;
            img[i] = (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        return (float)System.Math.Sqrt(sq / img.Length);
    }

    public static void FlipRows(byte[] src, byte[] dst, int w, int h)
    {
        int stride = w * 3;
        for (int y = 0; y < h; y++) System.Buffer.BlockCopy(src, y * stride, dst, (h - 1 - y) * stride, stride);
    }
}