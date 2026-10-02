using UnityEngine;

public class ConstantSpeedCar : MonoBehaviour
{
    public float speed = 10f; // m/s
    void FixedUpdate()
    {
        transform.position += transform.forward * speed * Time.fixedDeltaTime;
    }
}