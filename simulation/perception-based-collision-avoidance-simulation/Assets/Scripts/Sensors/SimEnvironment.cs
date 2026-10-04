// Environment properties that sensors read. The scenario manager (phase 8) will set them.
public static class SimEnvironment
{
    // meteorological visibility in meters; 0 (or less) means clear air
    public static float FogVisibilityM = 0f;

    // extinction coefficient in 1/m (Koschmieder: beta = 3.912 / V)
    public static float FogExtinction { get { return FogVisibilityM > 0f ? 3.912f / FogVisibilityM : 0f; } }
}