// Calibration for the processor. Processor convention throughout: x forward, y left, z up;
// vehicle frame origin = centre of the rear axle at ground level. Mount angles in degrees.
// It deliberately contains nothing about noise, dropout, fog or seeds: those are simulator internals.

public class EgoCalibration
{
    public float wheelbase;
    public float length, width, height;
    public float rearOverhang;    // rear axle to the rear end of the body
    public float frontOverhang;   // front axle to the front end of the body
}

public class OpticalTransform
{
    public double[] R = new double[9];   // row-major: p_optical = R * p_vehicle + t
    public double[] t = new double[3];
}

public class CameraCalibration
{
    public int width, height;
    public float rateHz;
    public string format;
    public float fx, fy, cx, cy;                 // pixels; pixel (0,0) is the top-left corner, pixel i covers [i, i+1)
    public float horizontalFovDeg, verticalFovDeg;
    public float[] distortion = new float[5];    // k1 k2 p1 p2 k3, all zero: ideal pinhole camera
    public Mount mount;
    public OpticalTransform opticalFromVehicle;  // camera optical frame: x right, y down, z forward
}

public class LidarCalibration
{
    public int rows, cols;
    public float rateHz, maxRangeM;
    public Mount mount;
    public float[] elevationsDeg;                // row 0 = highest beam
    public float[] azimuthsDeg;                  // column 0 = leftmost, positive to the left
    public float rangeUnitM = 0.01f;
    public int noReturnCode = 0;                 // dropout, fog, grazing: NOT free space
    public int maxRangeCode = 65535;             // the ray reached maximum range without a hit
}

public class CalibrationData
{
    public int schema = 1;
    public string calibrationId = "";
    public string conventions = "right-handed; x forward, y left, z up; origin at the rear axle centre at ground level; " +
        "mount angles in degrees: yaw positive left, pitch positive up, roll right-hand rule about forward, intrinsic order yaw-pitch-roll";
    public EgoCalibration ego;
    public CameraCalibration camera;
    public LidarCalibration lidar;
    public float poseRateHz;
}