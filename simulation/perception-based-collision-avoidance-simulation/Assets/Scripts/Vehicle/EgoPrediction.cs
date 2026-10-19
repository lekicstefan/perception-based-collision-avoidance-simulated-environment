using UnityEngine;

// Predicts when the ego car reaches a point of its route if nothing intervenes: the scripted driver (speed profile and its P controller)
// and the longitudinal model of the vehicle (actuation delay, jerk limit, acceleration limits), stepped at the physics rate on the arc length of the
// route. It is the "unbraked" schedule of the scenario: it does not depend on what the system under test does, so every system variant faces the
// same timing. It is used by the time-to-reach triggers of moving objects (WaypointMover).
//
// The order of the steps is the one of the scene: Vehicle (speed and position from the request of the step before), then the driver (new request).
// The time of a position is the simulation time of the step that reached it (the first step is at dt).
public static class EgoPrediction
{
    // Time (s, simulation time) at which the rear axle reaches route arc length sRear, starting at rest at the start of the route.
    // Throws when the point is not on the route, beyond where the driver stops, or never reached.
    public static float TimeToReach(RouteDriver driver, Vehicle vehicle, float sRear)
    {
        Route track = driver.Track;
        if (track == null) throw new System.InvalidOperationException("EgoPrediction: the driver has no route yet");
        if (sRear <= 0f) return 0f;
        float stopAt = track.Length - driver.endMargin;
        if (sRear >= stopAt)
            throw new System.ArgumentException("EgoPrediction: the point is at route " + sRear.ToString("F1") + " m, but the driver stops at " + stopAt.ToString("F1") + " m (route length " + track.Length.ToString("F1") + " m)");

        var profile = new SpeedProfile(track, driver.cruiseSpeed, driver.maxLateralAccel, driver.comfortDecel, driver.comfortAccel);
        float dt = Time.fixedDeltaTime;
        int n = Mathf.Max(0, Mathf.RoundToInt(vehicle.actuationDelay / dt));
        var delayLine = new float[n];
        int idx = 0;
        float requested = 0f, accel = 0f, speed = 0f, s = 0f;
        float maxJerkStep = vehicle.jerkLimit * dt;
        int maxSteps = Mathf.RoundToInt(600f / dt);

        for (int step = 1; step <= maxSteps; step++)
        {
            // Vehicle.FixedUpdate: delay, clamp, jerk limit, speed (trapezoid), distance
            float delayed = requested;
            if (n > 0)
            {
                delayed = delayLine[idx];
                delayLine[idx] = requested;
                idx = (idx + 1) % n;
            }
            float target = Mathf.Clamp(delayed, -vehicle.maxBrake, vehicle.maxAccel);
            float aPrev = accel;
            accel = aPrev + Mathf.Clamp(target - aPrev, -maxJerkStep, maxJerkStep);
            float vOld = speed;
            float vNew = vOld + 0.5f * (aPrev + accel) * dt;
            if (vNew <= 0f)
            {
                vNew = 0f;
                if (accel < 0f) accel = 0f;
            }
            speed = vNew;
            float ds = 0.5f * (vOld + vNew) * dt;
            float sPrev = s;
            s += ds;
            if (s >= sRear)
            {
                float f = ds > 0f ? (sRear - sPrev) / ds : 1f;
                return (step - 1 + f) * dt;
            }

            // RouteDriver.FixedUpdate: profile feed-forward plus P correction
            float vT, aFf;
            profile.At(s, out vT, out aFf);
            float aReq = aFf + driver.speedGain * (vT - speed);
            requested = Mathf.Clamp(aReq, -driver.driverMaxBrake, vehicle.maxAccel);
        }
        throw new System.ArgumentException("EgoPrediction: the scripted car does not reach route " + sRear.ToString("F1") + " m within 600 s");
    }
}