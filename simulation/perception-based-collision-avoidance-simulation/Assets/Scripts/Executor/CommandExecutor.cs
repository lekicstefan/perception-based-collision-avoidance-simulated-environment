using System.Collections.Concurrent;
using UnityEngine;

[DefaultExecutionOrder(5)] // after Vehicle (-100) and RouteDriver (0)
public class CommandExecutor : MonoBehaviour
{
    public enum Rule { Driver, Stop }

    public Vehicle vehicle;
    public RouteDriver driver;

    public Rule AppliedRule { get; private set; }
    public float AppliedAccel { get; private set; }
    public bool HasMessage { get; private set; }
    public long ObjectId { get; private set; }          // -1 without a message
    public float Brake { get; private set; }            // m/s^2, fixed when the message arrived
    public float Remaining { get; private set; }        // m to the stop position
    public bool Waiting { get; private set; }           // stopped with a message in force
    public float DataAge { get; private set; }          // sim time from the capture of the newest command's data to its application
    public int Ignored { get; private set; }            // commands that arrived after a newer one

    readonly ConcurrentQueue<VehicleCommand> incoming = new ConcurrentQueue<VehicleCommand>();
    double lastTimestamp = double.NegativeInfinity;
    StopMessage message;

    public void Submit(VehicleCommand c) { incoming.Enqueue(c); } // any thread

    void Start()
    {
        if (driver != null) driver.writeAccelToVehicle = false;
    }

    void FixedUpdate()
    {
        VehicleCommand tmp, latest = default(VehicleCommand);
        bool got = false;
        while (incoming.TryDequeue(out tmp))
        {
            if (tmp.dataTimestamp <= lastTimestamp || (got && tmp.dataTimestamp <= latest.dataTimestamp)) { Ignored++; continue; }
            latest = tmp;
            got = true;
        }

        float v = vehicle.Speed;
        if (got)
        {
            lastTimestamp = latest.dataTimestamp;
            DataAge = (float)(SimClock.Now - latest.dataTimestamp);
            Latency.CommandApplied(Latency.Key(latest.dataTimestamp), Latency.Ns());
            if (!latest.hasMessage)
            {
                HasMessage = false;
                ObjectId = -1;
                Brake = 0f;
            }
            else if (!HasMessage || latest.message.objectId != message.objectId || latest.message.stop != message.stop)
            {
                // a new or changed message: the constant deceleration that stops the car at the stop position, worked out once
                message = latest.message;
                HasMessage = true;
                ObjectId = message.objectId;
                // the car keeps its speed until the actuation delay has passed, so that distance is not left for braking
                float remaining = (float)(message.stop - vehicle.Odometer) - v * vehicle.actuationDelay;
                Brake = remaining > 0f ? Mathf.Min(v * v / (2f * remaining), vehicle.maxBrake) : vehicle.maxBrake;
            }
        }

        Remaining = HasMessage ? (float)(message.stop - vehicle.Odometer) : 0f;
        Waiting = HasMessage && v < 0.05f;

        // the system can only brake
        float aDrv = driver != null ? driver.DesiredAccel : 0f;
        float aOut = HasMessage ? Mathf.Min(aDrv, -Brake) : aDrv;
        AppliedRule = HasMessage ? Rule.Stop : Rule.Driver;
        AppliedAccel = aOut;
        vehicle.RequestedAccel = aOut;

        // while braking the driver does not wind up: its acceleration starts again from zero after the release
        if (HasMessage && driver != null) driver.ResetAccel();
    }
}