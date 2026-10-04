public class LidarScan
{
    public const ushort NoReturn = 0;               // dropout, fog, grazing: NOT free space
    public const ushort MaxRangeOrSky = 65535;      // the ray travelled the full range without a hit

    public double timestamp;   // simulation time of the capture step (s)
    public int frameId;
    public int rows, cols;
    public ushort[] ranges;    // centimetres, row-major, row 0 = highest beam, column 0 = leftmost
    public PoseState pose;     // ego state at the capture step
}