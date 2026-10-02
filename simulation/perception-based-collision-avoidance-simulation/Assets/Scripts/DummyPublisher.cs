using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public class DummyPublisher : MonoBehaviour
{
    [SerializeField] int port = 5555;
    [SerializeField] float rateHz = 10f;

    readonly ConcurrentQueue<byte[]> queue = new ConcurrentQueue<byte[]>();
    Thread thread;
    volatile bool running;
    double nextSendTime;
    int frameId;

    void Start()
    {
        UnityEngine.Debug.Log("DummyPublisher: Start");
        nextSendTime = Time.fixedTimeAsDouble;
        running = true;
        thread = new Thread(NetLoop) { IsBackground = true };
        thread.Start();
    }

    void FixedUpdate()
    {
        double t = Time.fixedTimeAsDouble;
        if (t + 1e-9 >= nextSendTime)
        {
            var msg = new byte[12];
            BitConverter.GetBytes(t).CopyTo(msg, 0);
            BitConverter.GetBytes(frameId++).CopyTo(msg, 8);
            queue.Enqueue(msg);
            nextSendTime += 1.0 / rateHz;
        }
    }

    void NetLoop()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();

            using (var pub = new PublisherSocket())
            {
                pub.Options.SendHighWatermark = 10;
                pub.Bind($"tcp://127.0.0.1:{port}");
                UnityEngine.Debug.Log($"DummyPublisher: bound on {port}");
                int sent = 0;
                while (running)
                {
                    if (queue.TryDequeue(out var m))
                    {
                        pub.SendFrame(m);
                        if (++sent % 10 == 0) UnityEngine.Debug.Log($"DummyPublisher: sent {sent}");
                    }
                    else Thread.Sleep(1);
                }
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("DummyPublisher thread failed: " + e);
        }
    }

    void OnDestroy()
    {
        running = false;
        thread?.Join(1000);
        NetMQConfig.Cleanup(false);
    }
}