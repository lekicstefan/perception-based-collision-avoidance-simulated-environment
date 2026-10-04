using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.IO;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using static System.Net.Mime.MediaTypeNames;
using Stopwatch = System.Diagnostics.Stopwatch;

public static class SimState
{
    public static volatile bool Running;
    public static volatile float BrakeDecel; // m/s^2, 0 = no braking
}

public class DummyPublisher : MonoBehaviour
{
    [SerializeField] int sensorPort = 5555;
    [SerializeField] int commandPort = 5556;
    [SerializeField] float rateHz = 10f;
    [SerializeField] ConstantSpeedCar car;

    const byte MsgHello = 1;
    const byte MsgState = 2;
    const byte CmdReady = 1;
    const byte CmdBrake = 2;
    const byte CmdEcho = 3;
    const int MaxFrames = 200000;

    readonly BlockingCollection<byte[]> queue = new BlockingCollection<byte[]>();
    readonly ConcurrentQueue<int> echoQueue = new ConcurrentQueue<int>();
    readonly long[] tEnq = new long[MaxFrames];
    readonly long[] tSent = new long[MaxFrames];
    readonly long[] tRecv = new long[MaxFrames];
    readonly long[] tApplied = new long[MaxFrames];

    Thread thread;
    volatile bool threadAlive;
    volatile bool readyReceived;
    bool started;
    double nextSendTime;
    int frameId;
    int maxFrameUsed = -1;
    long t0;

    void Start()
    {
        t0 = Stopwatch.GetTimestamp();
        SimState.Running = false;
        SimState.BrakeDecel = 0f;
        threadAlive = true;
        thread = new Thread(NetLoop) { IsBackground = true };
        thread.Start();
    }

    void FixedUpdate()
    {
        while (echoQueue.TryDequeue(out int eid))
            if (eid >= 0 && eid < MaxFrames) tApplied[eid] = Stopwatch.GetTimestamp();

        if (!started)
        {
            if (!readyReceived) return;
            started = true;
            SimState.Running = true;
            nextSendTime = Time.fixedTimeAsDouble;
            UnityEngine.Debug.Log("READY received, run started");
        }

        double t = Time.fixedTimeAsDouble;
        if (t + 1e-9 >= nextSendTime)
        {
            int fid = frameId++;
            if (fid < MaxFrames) { tEnq[fid] = Stopwatch.GetTimestamp(); maxFrameUsed = fid; }
            queue.Add(MakeMsg(MsgState, t, fid, car.transform.position.z, car.CurrentSpeed));
            nextSendTime += 1.0 / rateHz;
        }
    }

    // 21 bytes: type(1) sim_t(8) frame_id(4) z(4) speed(4), little-endian
    static byte[] MakeMsg(byte type, double t, int fid, float z, float speed)
    {
        var m = new byte[21];
        m[0] = type;
        BitConverter.GetBytes(t).CopyTo(m, 1);
        BitConverter.GetBytes(fid).CopyTo(m, 9);
        BitConverter.GetBytes(z).CopyTo(m, 13);
        BitConverter.GetBytes(speed).CopyTo(m, 17);
        return m;
    }

    void NetLoop()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();
            using (var pub = new PublisherSocket())
            using (var cmd = new SubscriberSocket())
            {
                pub.Options.SendHighWatermark = 10;
                pub.Bind($"tcp://127.0.0.1:{sensorPort}");
                cmd.Connect($"tcp://127.0.0.1:{commandPort}");
                cmd.SubscribeToAnyTopic();
                UnityEngine.Debug.Log("Sockets up");

                var sw = Stopwatch.StartNew();
                long lastHello = -1000;
                bool brakeLogged = false;

                while (threadAlive)
                {
                    while (cmd.TryReceiveFrameBytes(TimeSpan.Zero, out byte[] c))
                    {
                        long now = Stopwatch.GetTimestamp();
                        if (c.Length >= 1 && c[0] == CmdReady)
                        {
                            readyReceived = true;
                        }
                        else if (c.Length >= 9 && c[0] == CmdBrake)
                        {
                            uint id = BitConverter.ToUInt32(c, 1);
                            float decel = BitConverter.ToSingle(c, 5);
                            SimState.BrakeDecel = decel;
                            if (!brakeLogged)
                            {
                                brakeLogged = true;
                                UnityEngine.Debug.Log($"BRAKE received id={id} decel={decel}");
                            }
                        }
                        else if (c.Length >= 5 && c[0] == CmdEcho)
                        {
                            uint id = BitConverter.ToUInt32(c, 1);
                            if (id < MaxFrames)
                            {
                                tRecv[id] = now;
                                echoQueue.Enqueue((int)id);
                            }
                        }
                    }

                    if (!readyReceived && sw.ElapsedMilliseconds - lastHello >= 100)
                    {
                        lastHello = sw.ElapsedMilliseconds;
                        pub.SendFrame(MakeMsg(MsgHello, 0.0, 0, 0f, 0f));
                    }

                    if (queue.TryTake(out byte[] m, 1))
                    {
                        pub.SendFrame(m);
                        long ts = Stopwatch.GetTimestamp();
                        int fid = BitConverter.ToInt32(m, 9);
                        if (m[0] == MsgState && fid < MaxFrames) tSent[fid] = ts;
                    }
                }
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Net thread failed: " + e);
        }
    }

    string Ms(long ts)
    {
        if (ts == 0) return "";
        return ((ts - t0) * 1000.0 / Stopwatch.Frequency).ToString("F3", CultureInfo.InvariantCulture);
    }

    void SaveLatency()
    {
        if (maxFrameUsed < 0) return;
        string path = Path.Combine(UnityEngine.Application.persistentDataPath, "latency.csv");
        using (var w = new StreamWriter(path))
        {
            w.WriteLine("frame_id,t_enq_ms,t_sent_ms,t_recv_ms,t_applied_ms");
            for (int i = 0; i <= maxFrameUsed; i++)
                w.WriteLine($"{i},{Ms(tEnq[i])},{Ms(tSent[i])},{Ms(tRecv[i])},{Ms(tApplied[i])}");
        }
        UnityEngine.Debug.Log("Latency log written to " + path);
    }

    void OnDestroy()
    {
        threadAlive = false;
        thread?.Join(1000);
        NetMQConfig.Cleanup(false);
        SaveLatency();
    }
}