using UnityEngine;

public class CollisionTestScript : MonoBehaviour
{
    public Vehicle vehicle;
    public CommandExecutor executor;
    public BoxCollider egoBox;       // the CollisionBox collider
    public Collider hazard;          // an axis-aligned obstacle ahead along +Z
    public float triggerGap = 10f;   // m between the car's front and the obstacle when braking is commanded
    public bool sendEmergency = true;

    bool submitted, reported, expectCollision;
    float gap0, v0, expectedImpact, expectedFinalGap;

    float FrontZ()
    {
        Vector3 f = egoBox.transform.TransformPoint(egoBox.center + new Vector3(0f, 0f, 0.5f * egoBox.size.z));
        return f.z;
    }

    void FixedUpdate()
    {
        float gap = hazard.bounds.min.z - FrontZ();

        if (!submitted)
        {
            if (vehicle.Speed > 1f && gap <= triggerGap)
            {
                submitted = true;
                gap0 = gap;
                v0 = vehicle.Speed;

                float a = vehicle.maxBrake;
                float tr = a / vehicle.jerkLimit;
                float teff = vehicle.actuationDelay + 0.5f * tr;
                float dStop = v0 * teff + v0 * v0 / (2f * a) - a * tr * tr / 24f;
                if (sendEmergency)
                {
                    expectCollision = gap0 < dStop;
                    expectedImpact = expectCollision ? Mathf.Sqrt(2f * a * (dStop - gap0)) : 0f;
                    expectedFinalGap = gap0 - dStop;
                    executor.Submit(new VehicleCommand
                    {
                        id = 1,
                        dataTimestamp = Time.fixedTimeAsDouble,
                        comfortCap = 99f,
                        safetyCap = 99f,
                        comfortDecel = 3f,
                        emergency = true,
                        validity = 60f,
                        riskLevel = 1
                    });
                }
                else
                {
                    expectCollision = true;
                    expectedImpact = v0;
                }
                UnityEngine.Debug.Log("TEST: command " + (sendEmergency ? "EMERGENCY" : "none") + " at gap " +
                    gap0.ToString("F2") + " m, speed " + v0.ToString("F2") + " m/s | model stopping distance " +
                    dStop.ToString("F2") + " m | expected: " +
                    (expectCollision ? "collision at about " + expectedImpact.ToString("F2") + " m/s"
                                     : "stop with a final gap of " + expectedFinalGap.ToString("F2") + " m"));
            }
            return;
        }

        if (reported) return;
        if (RunControl.Ended && RunControl.Reason == RunControl.EndReason.Collision)
        {
            reported = true;
            UnityEngine.Debug.Log("TEST RESULT: collision, recorded impact speed " + RunControl.ImpactSpeed.ToString("F2") +
                " m/s | expected " + (expectCollision ? expectedImpact.ToString("F2") + " m/s" : "no collision"));
        }
        else if (!RunControl.Ended && vehicle.Speed <= 0f)
        {
            reported = true;
            UnityEngine.Debug.Log("TEST RESULT: stopped, final gap " + gap.ToString("F2") + " m | expected " +
                (expectCollision ? "a collision" : expectedFinalGap.ToString("F2") + " m"));
        }
    }
}