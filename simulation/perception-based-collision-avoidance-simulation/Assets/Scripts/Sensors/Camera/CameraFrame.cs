public class CameraFrame
{
    public double timestamp;   // simulation time of the capture step (s)
    public int frameId;
    public int width, height;
    public string format;      // "jpeg" or "raw"
    public byte[] data;        // JPEG file, or raw RGB (top row first, 3 bytes per pixel, sRGB)
    public float latencyMs;    // SIMULATOR DIAGNOSTIC ONLY (wall-clock capture to delivery). Never serialized; the boundary test (4.9) fails if it appears in a message.
    public PoseState pose;     // ego state at the capture step
}