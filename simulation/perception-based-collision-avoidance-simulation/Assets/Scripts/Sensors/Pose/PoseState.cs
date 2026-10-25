// Ego state as the processor receives it. Processor convention, relative to the pose at the start of the run:
// x forward, y left, z up (m); yaw positive left, pitch positive up (rad).
public struct PoseState
{
    public double timestamp;          // simulation time of the step the pose refers to
    public double x, y, z;            // m
    public double yaw, pitch, roll;   // rad, yaw wrapped to (-pi, pi]
    public double odometer;           // m, the car's own path length since the start of the run
    public float speed;               // m/s
    public float yawRate;             // rad/s, positive left
    public float steeringAngle;       // rad, positive left
}