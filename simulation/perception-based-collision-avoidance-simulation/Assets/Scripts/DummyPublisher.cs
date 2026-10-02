using System;
using System.Collections.Concurrent;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public static class SimState
{
    public static volatile bool Running;
}

public class DummyPublisher : MonoBehaviour
{
    [SerializeField] int sensorPort = 5555;
    [SerializeField] int commandPort = 5556;
    [SerializeField] float rateHz = 10f;

    const byte MsgHello = 1;
    const byte MsgState = 2;
    const byte CmdReady = 1;

    readonly BlockingCollection<byte[]> queue = new BlockingCollection<byte[]>();
    Thread thread;
    volatile bool threadAlive;
    volatile bool readyReceived;
    bool started;
    double nextSendTime;
    int frameId;

    void Start()
    {
        SimState.Running = false;
        threadAlive = true;
        thread = new Thread(NetLoop) { IsBackground = true };
        thread.Start();
    }

    void FixedUpdate()
    {
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
            queue.Add(MakeMsg(MsgState, t, frameId++));
            nextSendTime += 1.0 / rateHz;
        }
    }

    static byte[] MakeMsg(byte type, double t, int fid)
    {
        var m = new byte[13];
        m[0] = type;
        BitConverter.GetBytes(t).CopyTo(m, 1);
        BitConverter.GetBytes(fid).CopyTo(m, 9);
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

                var sw = System.Diagnostics.Stopwatch.StartNew();
                long lastHello = -1000;

                while (threadAlive)
                {
                    while (cmd.TryReceiveFrameBytes(TimeSpan.Zero, out byte[] c))
                        if (c.Length >= 1 && c[0] == CmdReady) readyReceived = true;

                    if (!readyReceived && sw.ElapsedMilliseconds - lastHello >= 100)
                    {
                        lastHello = sw.ElapsedMilliseconds;
                        pub.SendFrame(MakeMsg(MsgHello, 0.0, 0));
                    }

                    if (queue.TryTake(out byte[] m, 5)) pub.SendFrame(m);
                }
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Net thread failed: " + e);
        }
    }

    void OnDestroy()
    {
        threadAlive = false;
        thread?.Join(1000);
        NetMQConfig.Cleanup(false);
    }
}