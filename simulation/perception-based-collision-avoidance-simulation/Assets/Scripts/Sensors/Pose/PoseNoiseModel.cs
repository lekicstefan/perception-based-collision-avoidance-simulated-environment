// Bias (random walk) plus white noise for the reported x, y and yaw.
public class PoseNoiseModel
{
    readonly System.Random rng;
    readonly double posStd, yawStd, posWalk, yawWalk;   // m, rad, m/sqrt(s), rad/sqrt(s)
    double bx, by, byaw;
    bool hasSpare;
    double spare;

    public double BiasX { get { return bx; } }
    public double BiasY { get { return by; } }
    public double BiasYaw { get { return byaw; } }

    public PoseNoiseModel(PoseConfig cfg, int seed)
    {
        const double D = System.Math.PI / 180.0;
        rng = new System.Random(seed);
        posStd = cfg.positionNoiseStdM;
        yawStd = cfg.yawNoiseStdDeg * D;
        posWalk = cfg.positionBiasWalkMPerSqrtS;
        yawWalk = cfg.yawBiasWalkDegPerSqrtS * D;
    }

    double Gauss()
    {
        if (hasSpare) { hasSpare = false; return spare; }
        double u1 = System.Math.Max(rng.NextDouble(), 1e-12), u2 = rng.NextDouble();
        double r = System.Math.Sqrt(-2.0 * System.Math.Log(u1)), a = 2.0 * System.Math.PI * u2;
        spare = r * System.Math.Sin(a);
        hasSpare = true;
        return r * System.Math.Cos(a);
    }

    // advances the bias by one step and returns the total offsets (bias + white noise) for this step
    public void Step(double dt, out double dx, out double dy, out double dyaw)
    {
        double s = System.Math.Sqrt(dt);
        bx += Gauss() * posWalk * s;
        by += Gauss() * posWalk * s;
        byaw += Gauss() * yawWalk * s;
        dx = bx + Gauss() * posStd;
        dy = by + Gauss() * posStd;
        dyaw = byaw + Gauss() * yawStd;
    }
}