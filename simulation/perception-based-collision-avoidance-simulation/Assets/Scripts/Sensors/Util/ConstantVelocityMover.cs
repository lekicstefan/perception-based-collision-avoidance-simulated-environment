using UnityEngine;

public class ConstantVelocityMover : MonoBehaviour
{
    public Vector3 velocity = new Vector3(1.5f, 0f, 0f);   // m/s, Unity world coordinates

    void FixedUpdate() { transform.position += velocity * Time.fixedDeltaTime; }
}