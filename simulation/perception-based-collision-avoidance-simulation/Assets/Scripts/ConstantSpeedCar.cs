using UnityEngine;

public class ConstantSpeedCar : MonoBehaviour
{
    public float speed = 10f; // m/s
    public float CurrentSpeed => speed;

    void FixedUpdate()
    {
        if (!SimState.Running) return;

        float decel = SimState.BrakeDecel;
        if (decel > 0f)
            speed = Mathf.Max(0f, speed - decel * Time.fixedDeltaTime);

        transform.position += transform.forward * speed * Time.fixedDeltaTime;
    }
}