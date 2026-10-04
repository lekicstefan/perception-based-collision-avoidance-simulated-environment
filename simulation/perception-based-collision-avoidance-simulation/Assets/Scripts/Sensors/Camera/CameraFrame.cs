public class CameraFrame
{
    public double timestamp;   // simulation time of the capture step (s)
    public int frameId;
    public int width, height;
    public string format;      // "jpeg" or "raw"
    public byte[] data;        // JPEG file, or raw RGB (top row first, 3 bytes per pixel, sRGB)
    public float latencyMs;    // wall-clock time from capture to delivery (readback, processing, encoding)
    public PoseState pose;     // ego state at the capture step
}