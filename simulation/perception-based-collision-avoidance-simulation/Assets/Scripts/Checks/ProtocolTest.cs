using System.Text;
using UnityEngine;

// Checks Protocol.cs against the byte-exact test vectors of docs/protocol.md (section 8).
// Put it on an empty GameObject and press Play, or right-click the component and choose "Run protocol test".
// The result is in the Console. Remove the GameObject again afterwards.
public class ProtocolTest : MonoBehaviour
{
    const string PoseVectorHex =
        "41565053010006000700000000000000000000000000f83f0000000000802440000000000000e0bf" +
        "9a9999999999b93f9a9999999999b93f0000000000000000000000000000000000004841cdcc4c3d0ad7a33c00000000";
    const string CommandVectorHex =
        "41565043010002002c0000002a000000ae47e17a14aef73f33334b4100008c41000040400000000000002040" +
        "cdcc4c3ea69b443c00010000";

    int checks, failures;

    void Start() { Run(); }

    [ContextMenu("Run protocol test")]
    void Run()
    {
        checks = 0; failures = 0;

        // 1. POSE message must equal the Python test vector byte for byte
        var pose = new PoseState
        {
            timestamp = 1.5,
            x = 10.25,
            y = -0.5,
            z = 0.1,
            yaw = 0.1,
            pitch = 0.0,
            roll = 0.0,
            speed = 12.5f,
            yawRate = 0.05f,
            steeringAngle = 0.02f
        };
        byte[] poseMsg = Protocol.BuildPose(7, pose);
        Check(Hex(poseMsg) == PoseVectorHex, "POSE message equals the test vector", Hex(poseMsg));

        // 2. COMMAND test vector must parse to the right values
        Protocol.CmdType type; VehicleCommand c; string json, error;
        bool ok = Protocol.TryParseCommandMessage(FromHex(CommandVectorHex), out type, out c, out json, out error);
        Check(ok && type == Protocol.CmdType.Command, "COMMAND vector parses", error);
        Check(c.id == 42u && c.dataTimestamp == 1.48 && !c.emergency && c.riskLevel == 1, "COMMAND id, timestamp, flags", "");
        Check(Mathf.Approximately(c.comfortCap, 12.7f) && Mathf.Approximately(c.safetyCap, 17.5f) &&
              Mathf.Approximately(c.comfortDecel, 3.0f) && c.brakeRequest == 0f && Mathf.Approximately(c.validity, 2.5f) &&
              Mathf.Approximately(c.fallbackDecel, 0.2f) && Mathf.Approximately(c.processingTime, 0.012f),
              "COMMAND float fields", "");

        // 3. READY and TELEMETRY (hand-built messages, same layout as Python's build_ready / build_telemetry)
        ok = Protocol.TryParseCommandMessage(FromHex("415650430100010000000000"), out type, out c, out json, out error);
        Check(ok && type == Protocol.CmdType.Ready, "READY parses", error);
        byte[] tele = Encoding.UTF8.GetBytes("{\"speed\":1}");
        byte[] teleMsg = new byte[12 + tele.Length];
        System.Array.Copy(FromHex("41565043010003000b000000"), teleMsg, 12);
        System.Array.Copy(tele, 0, teleMsg, 12, tele.Length);
        ok = Protocol.TryParseCommandMessage(teleMsg, out type, out c, out json, out error);
        Check(ok && type == Protocol.CmdType.Telemetry && json == "{\"speed\":1}", "TELEMETRY parses", error);

        // 4. Malformed command messages are rejected
        byte[] bad = FromHex(CommandVectorHex);
        bad[0] = (byte)'X';
        Check(!Protocol.TryParseCommandMessage(bad, out type, out c, out json, out error), "bad magic rejected", error);
        bad = FromHex(CommandVectorHex);
        bad[4] = 2;
        Check(!Protocol.TryParseCommandMessage(bad, out type, out c, out json, out error), "other schema version rejected", error);
        bad = FromHex(CommandVectorHex);
        System.Array.Resize(ref bad, 40);
        Check(!Protocol.TryParseCommandMessage(bad, out type, out c, out json, out error), "truncated message rejected", error);

        // 5. Sizes of the data messages at the defaults
        var scan = new LidarScan { timestamp = 0.3, frameId = 3, rows = 32, cols = 600, ranges = new ushort[32 * 600], pose = pose };
        scan.ranges[0] = 0x1234; scan.ranges[1] = 65535;
        byte[] lidarMsg = Protocol.BuildLidar(3, scan);
        Check(lidarMsg.Length == 38492, "LIDAR message is 38492 bytes", lidarMsg.Length.ToString());
        Check(lidarMsg[88 + 4] == 0x34 && lidarMsg[88 + 5] == 0x12 && lidarMsg[88 + 6] == 0xFF && lidarMsg[88 + 7] == 0xFF,
              "LIDAR ranges are little-endian after the 4-byte prefix", "");

        var frame = new CameraFrame { timestamp = 0.1, frameId = 0, width = 640, height = 360, format = "raw", data = new byte[640 * 360 * 3], latencyMs = 99f, pose = pose };
        byte[] camMsg = Protocol.BuildCamera(0, frame);
        Check(camMsg.Length == 88 + 8 + 691200, "raw CAMERA message size", camMsg.Length.ToString());
        Check(camMsg[88 + 4] == 1, "CAMERA format byte is 1 for raw", "");

        // 6. Session messages and oracle
        Check(Encoding.UTF8.GetString(Protocol.BuildHello(0), 88, 12) == "{\"schema\":1}", "HELLO payload", "");
        byte[] eor = Protocol.BuildEndOfRun(2, 12.34);
        Check(Encoding.UTF8.GetString(eor, 88, eor.Length - 88) == "{\"sim_time\":12.34}", "END_OF_RUN payload", "");

        var oracle = new OracleFrame { timestamp = 0.3, pose = pose, objects = new[] { new OracleObject { id = 1, x = 10, length = 4.2f }, new OracleObject { id = 2 } } };
        Check(Protocol.BuildOracle(5, oracle).Length == 88 + 4 + 2 * 64, "ORACLE message size", "");

        string summary = "PROTOCOL TEST: " + checks + " checks, " + failures + " failures";
        if (failures == 0) UnityEngine.Debug.Log(summary + " - all good");
        else UnityEngine.Debug.LogError(summary);
    }

    void Check(bool ok, string name, string detail)
    {
        checks++;
        if (ok) return;
        failures++;
        UnityEngine.Debug.LogError("FAIL: " + name + (string.IsNullOrEmpty(detail) ? "" : "  [" + detail + "]"));
    }

    static string Hex(byte[] b)
    {
        var sb = new StringBuilder(b.Length * 2);
        foreach (byte x in b) sb.Append(x.ToString("x2"));
        return sb.ToString();
    }

    static byte[] FromHex(string s)
    {
        var b = new byte[s.Length / 2];
        for (int i = 0; i < b.Length; i++) b[i] = System.Convert.ToByte(s.Substring(2 * i, 2), 16);
        return b;
    }
}