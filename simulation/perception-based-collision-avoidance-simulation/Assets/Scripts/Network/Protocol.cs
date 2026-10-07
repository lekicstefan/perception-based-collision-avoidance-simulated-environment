using System;
using System.Globalization;
using System.Text;

// Wire format of docs/protocol.md (schema version 1). Everything is little-endian.
// Unity to Python: an 88-byte header followed by a payload. Python to Unity: a 12-byte header followed by a payload.
// This file only turns objects into bytes and bytes into objects. It has no sockets and no Unity calls,
// so it is safe to use from the network thread.
public static class Protocol
{
    public const ushort Version = 1;
    public const int HeaderSize = 88;
    public const int CommandHeaderSize = 12;
    public const int CommandPayloadSize = 44;
    public const int OracleRecordSize = 64;

    public const int PortSession = 5555;
    public const int PortCommand = 5556;
    public const int PortLidar = 5557;
    public const int PortCamera = 5558;
    public const int PortPose = 5559;
    public const int PortOracle = 5560;

    public enum MsgType : byte { Hello = 1, Calibration = 2, EndOfRun = 3, Lidar = 4, Camera = 5, Pose = 6, Oracle = 7 }
    public enum CmdType : byte { Ready = 1, Command = 2, Telemetry = 3 }

    // "AVP" followed by 'S' (sensor streams, Unity to Python) or 'C' (command stream, Python to Unity).
    const byte MagicA = (byte)'A', MagicV = (byte)'V', MagicP = (byte)'P';
    const byte MagicSensorLast = (byte)'S', MagicCommandLast = (byte)'C';

    // ------------------------------------------------------------------ byte helpers

    sealed class Writer
    {
        public readonly byte[] Buf;
        public int Pos;
        public Writer(int size) { Buf = new byte[size]; }

        public void U8(byte v) { Buf[Pos++] = v; }
        public void U16(ushort v) { Buf[Pos++] = (byte)v; Buf[Pos++] = (byte)(v >> 8); }
        public void U32(uint v) { for (int i = 0; i < 4; i++) Buf[Pos++] = (byte)(v >> (8 * i)); }
        public void U64(ulong v) { for (int i = 0; i < 8; i++) Buf[Pos++] = (byte)(v >> (8 * i)); }
        public void I32(int v) { U32(unchecked((uint)v)); }
        public void F32(float v) { U32(unchecked((uint)BitConverter.SingleToInt32Bits(v))); }
        public void F64(double v) { U64(unchecked((ulong)BitConverter.DoubleToInt64Bits(v))); }

        public void Bytes(byte[] src)
        {
            Buffer.BlockCopy(src, 0, Buf, Pos, src.Length);
            Pos += src.Length;
        }

        public void U16Array(ushort[] src)
        {
            if (BitConverter.IsLittleEndian)
            {
                Buffer.BlockCopy(src, 0, Buf, Pos, src.Length * 2);
                Pos += src.Length * 2;
            }
            else
            {
                for (int i = 0; i < src.Length; i++) U16(src[i]);
            }
        }
    }

    sealed class Reader
    {
        readonly byte[] b;
        public int Pos;
        public Reader(byte[] data) { b = data; }

        public byte U8() { return b[Pos++]; }
        public ushort U16() { ushort v = (ushort)(b[Pos] | (b[Pos + 1] << 8)); Pos += 2; return v; }
        public uint U32() { uint v = 0; for (int i = 0; i < 4; i++) v |= (uint)b[Pos + i] << (8 * i); Pos += 4; return v; }
        public ulong U64() { ulong v = 0; for (int i = 0; i < 8; i++) v |= (ulong)b[Pos + i] << (8 * i); Pos += 8; return v; }
        public float F32() { return BitConverter.Int32BitsToSingle(unchecked((int)U32())); }
        public double F64() { return BitConverter.Int64BitsToDouble(unchecked((long)U64())); }
    }

    // Starts a Unity-to-Python message: allocates header + payload and writes the 88-byte header.
    // The pose fields are the ego pose at the capture step of the message (all zero on session messages).
    static Writer NewMessage(MsgType type, uint seq, int payloadLength, double t, PoseState pose)
    {
        var w = new Writer(HeaderSize + payloadLength);
        w.U8(MagicA); w.U8(MagicV); w.U8(MagicP); w.U8(MagicSensorLast);
        w.U16(Version);
        w.U8((byte)type);
        w.U8(0);                          // flags, reserved
        w.U32(seq);
        w.U32((uint)payloadLength);
        w.F64(t);
        w.F64(pose.x); w.F64(pose.y); w.F64(pose.z);
        w.F64(pose.yaw); w.F64(pose.pitch); w.F64(pose.roll);
        w.F32(pose.speed); w.F32(pose.yawRate); w.F32(pose.steeringAngle);
        w.U32(0);                         // reserved (keeps the payload 8-byte aligned)
        if (w.Pos != HeaderSize) throw new InvalidOperationException("header size is not 88 bytes");
        return w;
    }

    // ------------------------------------------------------------------ Unity to Python: building

    public static byte[] BuildPose(uint seq, PoseState p)
    {
        return NewMessage(MsgType.Pose, seq, 0, p.timestamp, p).Buf;
    }

    public static byte[] BuildLidar(uint seq, LidarScan s)
    {
        int n = s.rows * s.cols;
        if (s.ranges == null || s.ranges.Length != n) throw new ArgumentException("range image does not match rows * cols");
        var w = NewMessage(MsgType.Lidar, seq, 4 + 2 * n, s.timestamp, s.pose);
        w.U16((ushort)s.rows);
        w.U16((ushort)s.cols);
        w.U16Array(s.ranges);
        return w.Buf;
    }

    public static byte[] BuildCamera(uint seq, CameraFrame f)
    {
        byte format;
        if (f.format == "jpeg") format = 0;
        else if (f.format == "raw") format = 1;
        else throw new ArgumentException("unknown camera format '" + f.format + "'");
        if (format == 1 && f.data.Length != f.width * f.height * 3) throw new ArgumentException("raw image size does not match width * height * 3");

        // f.latencyMs is a simulator diagnostic and is deliberately NOT part of the message.
        var w = NewMessage(MsgType.Camera, seq, 8 + f.data.Length, f.timestamp, f.pose);
        w.U16((ushort)f.width);
        w.U16((ushort)f.height);
        w.U8(format);
        w.U8(0); w.U8(0); w.U8(0);        // reserved
        w.Bytes(f.data);
        return w.Buf;
    }

    // Session messages carry a JSON payload; t and the pose fields are zero.
    static byte[] BuildJson(MsgType type, uint seq, string json)
    {
        byte[] body = Encoding.UTF8.GetBytes(json);
        var w = NewMessage(type, seq, body.Length, 0.0, default(PoseState));
        w.Bytes(body);
        return w.Buf;
    }

    public static byte[] BuildHello(uint seq, string runDir = null)
    {
        string json = "{\"schema\":" + Version.ToString(CultureInfo.InvariantCulture);
        if (runDir != null) json += ",\"run_dir\":\"" + runDir.Replace('\\', '/').Replace("\"", "\\\"") + "\"";
        return BuildJson(MsgType.Hello, seq, json + "}");
    }

    public static byte[] BuildCalibration(uint seq, string calibrationJson)
    {
        return BuildJson(MsgType.Calibration, seq, calibrationJson);
    }

    public static byte[] BuildEndOfRun(uint seq, double simTime)
    {
        return BuildJson(MsgType.EndOfRun, seq, "{\"sim_time\":" + simTime.ToString("R", CultureInfo.InvariantCulture) + "}");
    }

    // Oracle mode only. Header pose and t are those of the LiDAR scan the frame belongs to.
    public static byte[] BuildOracle(uint seq, OracleFrame o)
    {
        int n = o.objects.Length;
        var w = NewMessage(MsgType.Oracle, seq, 4 + OracleRecordSize * n, o.timestamp, o.pose);
        w.U32((uint)n);
        for (int i = 0; i < n; i++)
        {
            OracleObject ob = o.objects[i];
            w.I32(ob.id);
            w.F64(ob.x); w.F64(ob.y); w.F64(ob.z);
            w.F64(ob.yaw); w.F64(ob.vx); w.F64(ob.vy);
            w.F32(ob.length); w.F32(ob.width); w.F32(ob.height);
        }
        return w.Buf;
    }

    // ------------------------------------------------------------------ Python to Unity: parsing

    // Parses one message from the command socket. Returns false (with a reason in 'error') for anything that does not
    // follow docs/protocol.md; the caller logs the error and drops the message.
    // READY: nothing extra. COMMAND: 'cmd' is filled. TELEMETRY: 'json' is filled.
    public static bool TryParseCommandMessage(byte[] msg, out CmdType type, out VehicleCommand cmd, out string json, out string error)
    {
        type = 0; cmd = default(VehicleCommand); json = null; error = null;
        if (msg == null || msg.Length < CommandHeaderSize) { error = "message too short for a command header"; return false; }

        var r = new Reader(msg);
        if (r.U8() != MagicA || r.U8() != MagicV || r.U8() != MagicP || r.U8() != MagicCommandLast) { error = "bad magic"; return false; }
        ushort version = r.U16();
        if (version != Version) { error = "schema version " + version + ", expected " + Version; return false; }
        byte rawType = r.U8();
        r.U8();                           // reserved
        uint length = r.U32();
        if (msg.Length != CommandHeaderSize + (long)length) { error = "length mismatch"; return false; }
        if (rawType < (byte)CmdType.Ready || rawType > (byte)CmdType.Telemetry) { error = "unknown command message type " + rawType; return false; }
        type = (CmdType)rawType;

        switch (type)
        {
            case CmdType.Ready:
                if (length != 0) { error = "READY has no payload"; return false; }
                return true;

            case CmdType.Command:
                if (length != CommandPayloadSize) { error = "COMMAND payload has the wrong size"; return false; }
                cmd.id = r.U32();
                cmd.dataTimestamp = r.F64();
                cmd.comfortCap = r.F32();
                cmd.safetyCap = r.F32();
                cmd.comfortDecel = r.F32();
                cmd.brakeRequest = r.F32();
                cmd.validity = r.F32();
                cmd.fallbackDecel = r.F32();
                cmd.processingTime = r.F32();
                cmd.emergency = r.U8() != 0;
                cmd.riskLevel = r.U8();
                r.U16();                  // reserved
                return true;

            default:                      // Telemetry
                json = Encoding.UTF8.GetString(msg, CommandHeaderSize, (int)length);
                return true;
        }
    }
}