using UnityEngine;

// Processor convention (right-handed): x forward, y left, z up.  Unity: x right, y up, z forward (left-handed).
public static class Frames
{
    public static Vector3 ToUnity(float x, float y, float z) { return new Vector3(-y, z, x); }

    public static void FromUnity(Vector3 u, out float x, out float y, out float z) { x = u.z; y = -u.x; z = u.y; }

    // yaw left+, pitch up+, roll right-hand about forward -> Unity Euler angles (all three change sign)
    public static Quaternion MountRotationToUnity(float yawDeg, float pitchDeg, float rollDeg)
    {
        return Quaternion.Euler(-pitchDeg, -yawDeg, -rollDeg);
    }
}