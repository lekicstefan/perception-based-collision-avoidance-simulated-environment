// Captures at the fixed step nearest to the ideal sample times k / rate, without drift.
public class SensorSchedule
{
    readonly double period;
    double next;

    public SensorSchedule(float rateHz, double startTime = 0.0)
    {
        period = 1.0 / rateHz;
        next = startTime;
    }

    public bool Due(double now, double dt)
    {
        if (now + 0.5 * dt < next) return false;
        next += period;
        while (next <= now - 0.5 * dt) next += period; // never fall behind (only if rate > step rate)
        return true;
    }
}