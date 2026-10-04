using UnityEngine;

public class CalibrationProvider : MonoBehaviour
{
    public SensorRig rig;
    public Vehicle vehicle;
    public BoxCollider egoBox;      // the CollisionBox: its size gives the ego dimensions

    public CalibrationData Calibration { get; private set; }
    public string Json { get; private set; }    // compact; what the calibration message will carry

    void Start()
    {
        Calibration = CalibrationBuilder.Build(rig.Config, vehicle, egoBox);
        Json = CalibrationBuilder.ToJson(Calibration, false);
        EgoCalibration e = Calibration.ego;
        UnityEngine.Debug.Log("CALIBRATION id " + Calibration.calibrationId + " | " + Json.Length + " bytes of JSON | ego " +
            e.length.ToString("F2") + " x " + e.width.ToString("F2") + " x " + e.height.ToString("F2") + " m, wheelbase " + e.wheelbase.ToString("F2"));
    }
}