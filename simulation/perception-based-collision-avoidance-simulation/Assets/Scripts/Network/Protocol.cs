using System;
using System.Globalization;
using System.Text;

public static class Protocol
{
    public const ushort Version = 1;
    public const int HeaderSize = 88;
    public const int CommandHeaderSize = 12;
    public const int CommandPayloadSize = 44;
    public const int BundleTimingSize = 32;      // frame id, flags, capture / built / sent time stamps
    public const int BundleBuiltOffset = HeaderSize + 16, BundleSentOffset = HeaderSize + 24;
    public const byte BundleHasLidar = 1, BundleHasCamera = 2;
    public const int OracleRecordSize = 64;

    public const int PortSession = 5555;
    public const int PortCommand = 5556;
    public const int PortBundle = 5557;
    public const int PortOracle = 5560;

    public enum MsgType : byte { Hello = 1, Calibration = 2, EndOfRun = 3, Bundle = 4, Oracle = 7 }
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

    // One message per frame: the header carries the capture time and the pose of the frame; the payload is
    //   timing block (32 bytes): frame id u32, flags u8 (1 lidar, 2 camera), 3 reserved, capture ns i64, built ns i64, sent ns i64
    //   lidar section (if flag 1): rows u16, cols u16, ranges u16 x rows*cols
    //   camera section (if flag 2, runs to the end): width u16, height u16, format u8 (0 jpeg, 1 raw), 3 reserved, data
    // The time stamps are Stopwatch ticks in ns (the Windows performance counter, shared with Python's perf_counter_ns).
    // 'built' is written by the main thread when the message is complete, 'sent' by the network thread right before the message leaves (Stamp).
    public static byte[] BuildBundle(int frameId, LidarScan s, CameraFrame f, long captureNs)
    {
        int len = BundleTimingSize;
        if (s != null)
        {
            if (s.ranges == null || s.ranges.Length != s.rows * s.cols) throw new ArgumentException("range image does not match rows * cols");
            len += 4 + 2 * s.ranges.Length;
        }
        byte format = 0;
        if (f != null)
        {
            if (f.format == "jpeg") format = 0;
            else if (f.format == "raw") format = 1;
            else throw new ArgumentException("unknown camera format '" + f.format + "'");
            if (format == 1 && f.data.Length != f.width * f.height * 3) throw new ArgumentException("raw image size does not match width * height * 3");
            len += 8 + f.data.Length;
        }
        double t = s != null ? s.timestamp : f.timestamp;
        PoseState pose = s != null ? s.pose : f.pose;
        var w = NewMessage(MsgType.Bundle, (uint)frameId, len, t, pose);
        w.U32((uint)frameId);
        w.U8((byte)((s != null ? BundleHasLidar : 0) | (f != null ? BundleHasCamera : 0)));
        w.U8(0); w.U8(0); w.U8(0);
        w.U64((ulong)captureNs);
        w.U64(0);                         // built, see Stamp
        w.U64(0);                         // sent, see Stamp
        if (s != null)
        {
            w.U16((ushort)s.rows);
            w.U16((ushort)s.cols);
            w.U16Array(s.ranges);
        }
        if (f != null)
        {
            w.U16((ushort)f.width);
            w.U16((ushort)f.height);
            w.U8(format);
            w.U8(0); w.U8(0); w.U8(0);
            w.Bytes(f.data);
        }
        return w.Buf;
    }

    public static void Stamp(byte[] msg, int offset, long ns)
    {
        ulong v = (ulong)ns;
        for (int i = 0; i < 8; i++) msg[offset + i] = (byte)(v >> (8 * i));
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
                if (length > 0) json = Encoding.UTF8.GetString(msg, CommandHeaderSize, (int)length);
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