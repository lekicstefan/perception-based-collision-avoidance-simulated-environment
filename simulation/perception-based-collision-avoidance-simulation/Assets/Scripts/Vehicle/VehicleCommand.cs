// Command from the processor to the vehicle. Python decides; Unity only enforces.
public struct VehicleCommand
{
    public uint id;
    public double dataTimestamp;  // sim time of the newest data used (s)
    public float comfortCap;      // m/s, held in normal operation
    public float safetyCap;       // m/s, hard limit
    public float comfortDecel;    // m/s^2, deceleration limit when slowing toward the comfort cap
    public float brakeRequest;    // m/s^2 (>= 0), 0 = none
    public bool emergency;        // full emergency braking
    public float validity;        // s, counted from the step Unity applies the command
    public float fallbackDecel;   // m/s^2 applied after expiry until stopped, 0 = release
    public byte riskLevel;        // 0 low, 1 medium, 2 high (display and log only)
    public float processingTime;  // s, measured by the processor (log only)
}