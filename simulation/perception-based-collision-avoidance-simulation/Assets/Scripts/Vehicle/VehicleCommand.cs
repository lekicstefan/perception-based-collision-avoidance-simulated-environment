public struct StopMessage
{
    public uint objectId;
    public double stop;      // m, odometer reading the car must stand still before
}

public struct VehicleCommand
{
    public double dataTimestamp;  // sim time of the capture the command is based on; a newer command replaces an older one
    public bool hasMessage;       // false: nothing threatens the car
    public StopMessage message;
}