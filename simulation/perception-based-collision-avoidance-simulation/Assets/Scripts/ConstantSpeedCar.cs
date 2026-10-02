using UnityEngine;

public class ConstantSpeedCar : MonoBehaviour
{
    public float speed = 10f; // m/s
    void FixedUpdate()
    {
        if (!SimState.Running) return;
        transform.position += transform.forward * speed * Time.fixedDeltaTime;
    }
}