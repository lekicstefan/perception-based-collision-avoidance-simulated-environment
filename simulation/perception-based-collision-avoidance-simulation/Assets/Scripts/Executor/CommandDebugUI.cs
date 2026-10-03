using System.Globalization;
using UnityEngine;

public class CommandDebugUI : MonoBehaviour
{
    public CommandExecutor executor;
    public Vehicle vehicle;

    string sComfortCap = "99", sSafetyCap = "99", sComfortDecel = "3", sBrake = "0";
    string sValidity = "3", sFallback = "0", sRisk = "1";
    bool emergency;
    uint nextId = 1;

    static float F(string s, float fallback)
    {
        float v;
        return float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v) ? v : fallback;
    }

    string Field(string label, string value)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(label, GUILayout.Width(170));
        value = GUILayout.TextField(value, GUILayout.Width(80));
        GUILayout.EndHorizontal();
        return value;
    }

    void Send(bool forceEmergency)
    {
        var c = new VehicleCommand
        {
            id = nextId++,
            dataTimestamp = Time.fixedTimeAsDouble,
            comfortCap = F(sComfortCap, 99f),
            safetyCap = F(sSafetyCap, 99f),
            comfortDecel = F(sComfortDecel, 3f),
            brakeRequest = Mathf.Max(0f, F(sBrake, 0f)),
            emergency = forceEmergency,
            validity = Mathf.Max(0.01f, F(sValidity, 3f)),
            fallbackDecel = Mathf.Max(0f, F(sFallback, 0f)),
            riskLevel = (byte)Mathf.Clamp(Mathf.RoundToInt(F(sRisk, 1f)), 0, 2),
            processingTime = 0f
        };
        executor.Submit(c);
    }

    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 40, 340, 440), GUI.skin.box);
        GUILayout.Label("COMMAND (decimals with a dot)");
        sComfortCap = Field("Comfort cap (m/s)", sComfortCap);
        sSafetyCap = Field("Safety cap (m/s)", sSafetyCap);
        sComfortDecel = Field("Comfort decel (m/s^2)", sComfortDecel);
        sBrake = Field("Brake request (m/s^2)", sBrake);
        sValidity = Field("Validity (s)", sValidity);
        sFallback = Field("Fallback decel (m/s^2)", sFallback);
        sRisk = Field("Risk level (0-2)", sRisk);
        emergency = GUILayout.Toggle(emergency, "Emergency flag");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Send")) Send(emergency);
        if (GUILayout.Button("Emergency now")) Send(true);
        if (GUILayout.Button("Clear")) executor.Clear();
        GUILayout.EndHorizontal();

        GUILayout.Space(8);
        GUILayout.Label("STATUS");
        string remaining = executor.Status == CommandExecutor.CommandStatus.Active
            ? "  (remaining " + executor.Remaining.ToString("F1") + " s)" : "";
        GUILayout.Label("State: " + executor.Status + remaining + "   last id " + executor.LastId);
        GUILayout.Label("Rule in force: " + executor.AppliedRule);
        GUILayout.Label("Speed " + vehicle.Speed.ToString("F2") + " m/s | accel " + vehicle.Accel.ToString("F2") +
                        " | requested " + executor.AppliedAccel.ToString("F2") + " m/s^2");
        GUILayout.EndArea();
    }
}