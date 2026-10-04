using UnityEngine;

public class EnvironmentSettings : MonoBehaviour
{
    public float fogVisibilityM = 0f;   // 0 = clear. Typical test values: 100, 200, 500.

    void Awake() { SimEnvironment.FogVisibilityM = fogVisibilityM; }
}