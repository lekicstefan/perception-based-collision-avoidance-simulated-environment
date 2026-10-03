using System.Collections.Concurrent;
using UnityEngine;

// Mechanical command executor. The processor decides; this only enforces.
// The system can only brake: the final acceleration is never above the driver's.
[DefaultExecutionOrder(5)] // after Vehicle (-100) and RouteDriver (0)
public class CommandExecutor : MonoBehaviour
{
    public enum CommandStatus { NoCommand, Active, Expired }
    public enum Rule { Driver, ComfortCap, SafetyCap, BrakeRequest, Emergency, Fallback, Released }

    public Vehicle vehicle;
    public RouteDriver driver;
    public float capGain = 1f;          // 1/s, P gain when following a cap
    public bool enforceExpiry = true;   // off = ablation "expiring commands off"
    public bool logCommands = true;     // turn off when commands arrive at 30 Hz

    public CommandStatus Status { get; private set; }
    public Rule AppliedRule { get; private set; }
    public uint LastId { get; private set; }
    public float AppliedAccel { get; private set; }
    public float Remaining { get; private set; }
    public double AppliedAt { get; private set; }
    public double ExpiredAt { get; private set; }
    public double ExpiredX { get; private set; }
    public double ExpiredZ { get; private set; }

    readonly ConcurrentQueue<VehicleCommand> incoming = new ConcurrentQueue<VehicleCommand>();
    VehicleCommand active;

    public void Submit(VehicleCommand c) { incoming.Enqueue(c); } // any thread
    public void Clear() { Status = CommandStatus.NoCommand; }     // main thread

    void Start()
    {
        if (driver != null) driver.writeAccelToVehicle = false;
    }

    void FixedUpdate()
    {
        double now = Time.fixedTimeAsDouble;

        // 1. the newest submitted command wins
        VehicleCommand tmp, latest = default(VehicleCommand);
        bool got = false;
        while (incoming.TryDequeue(out tmp)) { latest = tmp; got = true; }
        if (got)
        {
            active = latest;
            Status = CommandStatus.Active;
            AppliedAt = now;
            LastId = latest.id;
            if (logCommands)
                UnityEngine.Debug.Log("COMMAND #" + latest.id + " applied | comfort cap " + latest.comfortCap.ToString("F1") +
                    " | safety cap " + latest.safetyCap.ToString("F1") + " | comfort decel " + latest.comfortDecel.ToString("F1") +
                    " | brake " + latest.brakeRequest.ToString("F1") + " | emergency " + latest.emergency +
                    " | validity " + latest.validity.ToString("F2") + " s | fallback " + latest.fallbackDecel.ToString("F1"));
        }

        // 2. expiry in simulation time (half a step of tolerance against float rounding)
        if (Status == CommandStatus.Active)
        {
            double age = now - AppliedAt;
            Remaining = (float)System.Math.Max(0.0, active.validity - age);
            if (enforceExpiry && age >= active.validity - 0.5 * Time.fixedDeltaTime)
            {
                Status = CommandStatus.Expired;
                ExpiredAt = now;
                ExpiredX = vehicle.X;
                ExpiredZ = vehicle.Z;
                if (logCommands)
                    UnityEngine.Debug.Log("COMMAND #" + active.id + " EXPIRED after " + age.ToString("F2") +
                        " s | fallback decel " + active.fallbackDecel.ToString("F1"));
            }
        }

        // 3. merge with the driver (the system can only brake)
        float aDrv = driver != null ? driver.DesiredAccel : 0f;
        float aOut = aDrv;
        Rule rule = Rule.Driver;
        float v = vehicle.Speed;

        if (Status == CommandStatus.Expired)
        {
            if (active.fallbackDecel > 0f)
            {
                aOut = -Mathf.Min(active.fallbackDecel, vehicle.maxBrake);
                rule = Rule.Fallback;
            }
            else rule = Rule.Released;
        }
        else if (Status == CommandStatus.Active)
        {
            if (active.emergency)
            {
                aOut = -vehicle.maxBrake;
                rule = Rule.Emergency;
            }
            else if (active.brakeRequest > 0f)
            {
                aOut = -Mathf.Min(active.brakeRequest, vehicle.maxBrake);
                rule = Rule.BrakeRequest;
            }
            else
            {
                float aComf = Mathf.Clamp(capGain * (active.comfortCap - v), -Mathf.Max(0f, active.comfortDecel), vehicle.maxAccel);
                float aSafe = Mathf.Clamp(capGain * (active.safetyCap - v), -vehicle.maxBrake, vehicle.maxAccel);
                float lim = Mathf.Min(aComf, aSafe);
                if (aDrv > lim)
                {
                    aOut = lim;
                    rule = (aComf <= aSafe) ? Rule.ComfortCap : Rule.SafetyCap;
                }
            }
        }

        AppliedRule = rule;
        AppliedAccel = aOut;
        vehicle.RequestedAccel = aOut;
    }
}