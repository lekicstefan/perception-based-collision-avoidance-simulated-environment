using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

// Sensor configuration. Mount poses use the processor convention (right-handed), relative to the
// Vehicle root (rear axle centre at ground level): x forward, y left, z up (meters);
// yawDeg positive to the left, pitchDeg positive up, rollDeg by the right-hand rule about the forward axis.
// Partial JSON files are fine: every field that is missing keeps the default set in this file.

public class Mount
{
    public float x, y, z;
    public float yawDeg, pitchDeg, rollDeg;
}

public class ElevationGroup
{
    public int count = 1;
    public float fromDeg;
    public float toDeg;
    public bool includeFrom = true;
    public bool includeTo = true;

    // count beams, evenly spaced between fromDeg and toDeg; an excluded end keeps the spacing but is not a beam
    public float[] Generate()
    {
        if (count < 1) throw new ArgumentException("Sensor config: elevation group count must be >= 1");
        var r = new float[count];
        int intervals = count - 1 + (includeFrom ? 0 : 1) + (includeTo ? 0 : 1);
        if (intervals == 0) { r[0] = fromDeg; return r; }
        float step = (toDeg - fromDeg) / intervals;
        int k0 = includeFrom ? 0 : 1;
        for (int j = 0; j < count; j++) r[j] = fromDeg + (j + k0) * step;
        return r;
    }
}

public class PostProcessingConfig
{
    public bool noiseEnabled = false;
    public float noiseStd = 3f;          // intensity levels (0..255)
    public bool blurEnabled = false;
    public float blurSigmaPx = 1.5f;
    public bool exposureEnabled = false;
    public float exposureGain = 1f;      // 1 = unchanged, below 1 darkens (low-light scenario)
}

public class CameraConfig
{
    public bool enabled = true;
    public int width = 640;
    public int height = 360;
    public float rateHz = 30f;
    public float horizontalFovDeg = 90f;
    public Mount mount = new Mount { x = 1.8f, y = 0f, z = 1.3f };
    public string format = "jpeg";       // "jpeg" or "raw"
    public int jpegQuality = 85;
    public PostProcessingConfig postProcessing = new PostProcessingConfig();

    // pinhole intrinsics (square pixels). Pixel coordinates: pixel i covers [i, i+1), so the centre is width/2.
    [JsonIgnore] public float FocalPx { get { return (width * 0.5f) / Mathf.Tan(0.5f * horizontalFovDeg * Mathf.Deg2Rad); } }
    [JsonIgnore] public float Cx { get { return width * 0.5f; } }
    [JsonIgnore] public float Cy { get { return height * 0.5f; } }
    [JsonIgnore] public float VerticalFovDeg { get { return 2f * Mathf.Atan((height * 0.5f) / FocalPx) * Mathf.Rad2Deg; } }
}

public class LidarConfig
{
    public bool enabled = true;
    public Mount mount = new Mount { x = 1.3f, y = 0f, z = 1.8f };

    // explicit table wins if non-empty; otherwise the table is generated from the groups
    [JsonProperty(ObjectCreationHandling = ObjectCreationHandling.Replace)]
    public float[] elevationsDeg = new float[0];

    [JsonProperty(ObjectCreationHandling = ObjectCreationHandling.Replace)]
    public ElevationGroup[] elevationGroups = new[]
    {
        new ElevationGroup { count = 12, fromDeg = -20f, toDeg = -4f, includeFrom = true,  includeTo = false },
        new ElevationGroup { count = 16, fromDeg = -4f,  toDeg = 4f,  includeFrom = true,  includeTo = true  },
        new ElevationGroup { count = 4,  fromDeg = 4f,   toDeg = 10f, includeFrom = false, includeTo = true  }
    };

    public float horizontalFovDeg = 120f;
    public float horizontalResolutionDeg = 0.2f;
    public float maxRangeM = 100f;
    public float rateHz = 10f;
    public float rangeNoiseStdM = 0.02f;
    public float dropoutProbability = 0.01f;
    public bool grazingDropoutEnabled = false;
    public float grazingStartDeg = 85f;   // incidence angle where dropout starts
    public float grazingEndDeg = 90f;     // incidence angle where every return is dropped
    public float fogSensitivity = 1f;     // scales the fog attenuation (the scenario sets the fog)

    [JsonIgnore] public float[] ElevationsDeg { get; private set; }   // resolved: row 0 = highest beam
    [JsonIgnore] public int Rows { get { return ElevationsDeg.Length; } }
    [JsonIgnore] public int Columns { get; private set; }

    // azimuth of a column centre, positive to the left; column 0 is the leftmost
    public float AzimuthDeg(int col) { return horizontalFovDeg * 0.5f - (col + 0.5f) * horizontalResolutionDeg; }

    public void Resolve()
    {
        var list = new List<float>();
        if (elevationsDeg != null && elevationsDeg.Length > 0) list.AddRange(elevationsDeg);
        else foreach (ElevationGroup g in elevationGroups) list.AddRange(g.Generate());
        list.Sort();
        list.Reverse();
        ElevationsDeg = list.ToArray();

        Columns = Mathf.RoundToInt(horizontalFovDeg / horizontalResolutionDeg);
        if (Mathf.Abs(Columns * horizontalResolutionDeg - horizontalFovDeg) > 1e-3f)
            throw new ArgumentException("Sensor config: lidar.horizontalFovDeg must be a multiple of lidar.horizontalResolutionDeg");
    }
}

public class PoseConfig
{
    public float rateHz = 50f;                     // standalone state message rate
    public bool noiseEnabled = false;
    public float positionNoiseStdM = 0.05f;        // white noise per message
    public float yawNoiseStdDeg = 0.1f;
    public float positionBiasWalkMPerSqrtS = 0.02f;  // slowly varying bias: random walk
    public float yawBiasWalkDegPerSqrtS = 0.02f;
}

public class SensorConfig
{
    public int schema = 1;
    public string name = "default";
    public int seed = 12345;
    public CameraConfig camera = new CameraConfig();
    public LidarConfig lidar = new LidarConfig();
    public PoseConfig pose = new PoseConfig();

    static void Require(bool ok, string message)
    {
        if (!ok) throw new ArgumentException("Sensor config '" + "': " + message);
    }

    public void Resolve()
    {
        lidar.Resolve();
        CameraConfig c = camera;
        LidarConfig l = lidar;
        PoseConfig p = pose;
        Require(c.width > 0 && c.height > 0, "camera.width and camera.height must be > 0");
        Require(c.rateHz > 0f, "camera.rateHz must be > 0");
        Require(c.horizontalFovDeg > 10f && c.horizontalFovDeg < 170f, "camera.horizontalFovDeg must be between 10 and 170");
        Require(c.format == "jpeg" || c.format == "raw", "camera.format must be \"jpeg\" or \"raw\"");
        Require(c.jpegQuality >= 1 && c.jpegQuality <= 100, "camera.jpegQuality must be 1..100");
        Require(c.postProcessing.noiseStd >= 0f && c.postProcessing.blurSigmaPx >= 0f && c.postProcessing.exposureGain > 0f,
                "camera.postProcessing values out of range");
        Require(l.Rows >= 1, "lidar needs at least one beam");
        foreach (float e in l.ElevationsDeg) Require(e > -90f && e < 90f, "lidar elevations must be between -90 and 90 degrees");
        Require(l.horizontalResolutionDeg > 0f && l.horizontalFovDeg > 0f && l.horizontalFovDeg <= 360f, "lidar horizontal fov/resolution out of range");
        Require(l.maxRangeM > 0f && l.rateHz > 0f, "lidar.maxRangeM and lidar.rateHz must be > 0");
        Require(l.dropoutProbability >= 0f && l.dropoutProbability <= 1f, "lidar.dropoutProbability must be 0..1");
        Require(l.rangeNoiseStdM >= 0f && l.fogSensitivity >= 0f, "lidar noise/fog values must be >= 0");
        Require(l.grazingStartDeg < l.grazingEndDeg, "lidar.grazingStartDeg must be below grazingEndDeg");
        Require(p.rateHz > 0f && p.positionNoiseStdM >= 0f && p.yawNoiseStdDeg >= 0f, "pose values out of range");
    }
}