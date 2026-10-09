using System;
using System.Collections.Concurrent;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using Newtonsoft.Json.Linq;
using Stopwatch = System.Diagnostics.Stopwatch;

// Unity side of docs/protocol.md: sockets, handshake, heartbeat and shutdown.
//
// Threads: the main thread serializes sensor events into byte arrays (Protocol.Build*) and puts them in a queue.
// One background thread owns every NetMQ socket: it sends from the queue, receives commands, runs the handshake and
// sends the calibration heartbeat. Unity API calls stay on the main thread, sockets stay on the network thread.
//
// Run start: while waitForProcessor is on, Time.timeScale is 0 from Awake until the processor answers READY, so no
// FixedUpdate runs (no sensors, no driving, no logging). Then timeScale becomes 1 and the run starts with seq 0.
public class NetworkHost : MonoBehaviour
{
    [Header("Scene wiring")]
    public LidarSensor lidar;
    public CameraSensor cameraSensor;
    public PoseEstimator pose;
    public CalibrationProvider calibration;
    public CommandExecutor executor;
    public OracleFeed oracle;           // optional; the oracle socket exists only when oracle.oracleMode is on

    [Header("Run")]
    [Tooltip("On: the simulation stays frozen until the Python processor answers READY. Off: no sockets at all, for development without Python.")]
    public bool waitForProcessor = true;

    // socket indices
    const int SSession = 0, SLidar = 1, SCamera = 2, SPose = 3, SOracle = 4;
    static readonly int[] Hwm = { 1000, 5, 10, 100, 5 };                       // docs/protocol.md section 1
    static readonly int[] Ports = { Protocol.PortSession, Protocol.PortLidar, Protocol.PortCamera, Protocol.PortPose, Protocol.PortOracle };
    static readonly string[] Names = { "session", "lidar", "camera", "pose", "oracle" };
    const int SocketBufferBytes = 64 * 1024;    // bounds how stale queued data can get (see the stall experiment)
    const int MaxQueued = 500;                  // safety net, the network thread normally keeps the queue near empty

    struct Outgoing { public int socket; public byte[] data; }

    readonly BlockingCollection<Outgoing> outgoing = new BlockingCollection<Outgoing>();
    Thread thread;
    volatile bool threadAlive;
    volatile bool readyReceived;
    volatile bool netFailed;
    volatile bool endRequested;
    volatile string calibrationJson;
    volatile string latestTelemetry;
    volatile string readyJson;
    double endSimTime;
    CommandExecutor executorRef;                // read on the network thread, compared with ReferenceEquals (no Unity calls)

    bool started, failureReported;
    uint lidarSeq, cameraSeq, poseSeq, oracleSeq;   // main thread only
    readonly long[] sentCount = new long[5];        // network thread writes, read after Join
    int commandsReceived, parseErrors, queueDrops;

    public bool Started { get { return started; } }
    public string LatestTelemetry { get { return latestTelemetry; } }   // JSON for the dashboard (step 7.14)

    // Called by whoever ends the run (the scenario manager in phase 8). Sends END_OF_RUN to the processor.
    public void RequestEndOfRun(double simTime)
    {
        if (!started || endRequested)
        {
            RunFolder.Configure(null);
            return;
        }

        endSimTime = simTime;
        endRequested = true;
    }

    void Awake()
    {
        if (!waitForProcessor) { enabled = false; return; }
        Time.timeScale = 0f;
    }

    void OnEnable()
    {
        if (!waitForProcessor) return;
        if (lidar != null) lidar.ScanReady += OnScan;
        if (cameraSensor != null) cameraSensor.FrameReady += OnFrame;
        if (pose != null) pose.PoseReady += OnPose;
        if (oracle != null && oracle.oracleMode) oracle.OracleReady += OnOracle;
    }

    void OnDisable()
    {
        if (lidar != null) lidar.ScanReady -= OnScan;
        if (cameraSensor != null) cameraSensor.FrameReady -= OnFrame;
        if (pose != null) pose.PoseReady -= OnPose;
        if (oracle != null) oracle.OracleReady -= OnOracle;
    }

    void Start()
    {
        if (!waitForProcessor) return;
        if (calibration == null) UnityEngine.Debug.LogWarning("NetworkHost: no CalibrationProvider assigned, the processor will never get a calibration.");
        executorRef = executor;
        bool oracleMode = oracle != null && oracle.oracleMode;
        threadAlive = true;
        thread = new Thread(() => NetLoop(oracleMode)) { IsBackground = true, Name = "NetworkHost" };
        thread.Start();
    }

    void Update()
    {
        // CalibrationProvider builds its JSON in its own Start, so pick it up here once it exists.
        if (calibrationJson == null && calibration != null && calibration.Json != null) calibrationJson = calibration.Json;

        if (netFailed && !failureReported)
        {
            failureReported = true;
            UnityEngine.Debug.LogError("NetworkHost: the network thread failed (see the error above, often a port already in use). The simulation stays frozen.");
        }

        if (!started && readyReceived)
        {
            string path = null;
            if (!string.IsNullOrEmpty(readyJson)) path = (string)JObject.Parse(readyJson)["run_dir"];
            RunFolder.Configure(path);

            started = true;
            Time.timeScale = 1f;
            UnityEngine.Debug.Log("NetworkHost: READY received, the run starts now");
        }
    }

    // ------------------------------------------------------------------ main thread: sensor events to the queue

    void OnScan(LidarScan s) { if (started) Enqueue(SLidar, Protocol.BuildLidar(lidarSeq++, s)); }
    void OnFrame(CameraFrame f) { if (started) Enqueue(SCamera, Protocol.BuildCamera(cameraSeq++, f)); }
    void OnPose(PoseState p) { if (started) Enqueue(SPose, Protocol.BuildPose(poseSeq++, p)); }
    void OnOracle(OracleFrame o) { if (started) Enqueue(SOracle, Protocol.BuildOracle(oracleSeq++, o)); }

    void Enqueue(int socket, byte[] data)
    {
        if (outgoing.Count >= MaxQueued)
        {
            if (Interlocked.Increment(ref queueDrops) == 1) UnityEngine.Debug.LogWarning("NetworkHost: send queue is full, dropping messages");
            return;
        }
        outgoing.Add(new Outgoing { socket = socket, data = data });
    }

    // ------------------------------------------------------------------ network thread

    void NetLoop(bool oracleMode)
    {
        var pubs = new PublisherSocket[5];
        SubscriberSocket cmd = null;
        try
        {
            AsyncIO.ForceDotNet.Force();       // needed for NetMQ under Mono in the player (found in phase 1)
            int n = oracleMode ? 5 : 4;        // the oracle socket does not exist in normal mode
            for (int i = 0; i < n; i++)
            {
                var p = new PublisherSocket();
                p.Options.SendHighWatermark = Hwm[i];
                p.Options.Linger = TimeSpan.FromMilliseconds(500);
                if (i != SSession) p.Options.SendBuffer = SocketBufferBytes;
                p.Bind("tcp://127.0.0.1:" + Ports[i]);
                pubs[i] = p;
            }
            cmd = new SubscriberSocket();
            cmd.Options.ReceiveHighWatermark = 10;
            cmd.Connect("tcp://127.0.0.1:" + Protocol.PortCommand);
            cmd.SubscribeToAnyTopic();
            UnityEngine.Debug.Log("NetworkHost: sockets up (" + n + " streams" + (oracleMode ? ", ORACLE MODE" : "") + "), sending HELLO");
            Loop(pubs, cmd);
        }
        catch (Exception e)
        {
            netFailed = true;
            UnityEngine.Debug.LogError("NetworkHost: network thread failed: " + e);
        }
        finally
        {
            foreach (var p in pubs) if (p != null) p.Dispose();
            if (cmd != null) cmd.Dispose();
        }
    }

    void Loop(PublisherSocket[] pubs, SubscriberSocket cmd)
    {
        var clock = Stopwatch.StartNew();
        long lastHello = -1000, lastCalibration = -100000, lastEnd = -1000;
        uint helloSeq = 0, calibrationSeq = 0, endSeq = 0;
        int endSent = 0;

        while (threadAlive)
        {
            // 1. commands from the processor
            byte[] raw;
            while (cmd.TryReceiveFrameBytes(TimeSpan.Zero, out raw)) HandleCommandMessage(raw);

            long now = clock.ElapsedMilliseconds;

            if (!readyReceived)
            {
                // 2. HELLO every 100 ms until READY
                if (now - lastHello >= 100)
                {
                    lastHello = now;
                    pubs[SSession].SendFrame(Protocol.BuildHello(helloSeq++));
                }
            }
            else
            {
                // 3. calibration right after READY, then every second. It also serves as heartbeat while the simulation is paused.
                string json = calibrationJson;
                if (json != null && now - lastCalibration >= 1000)
                {
                    lastCalibration = now;
                    pubs[SSession].SendFrame(Protocol.BuildCalibration(calibrationSeq++, json));
                }
            }

            // 4. sensor data
            int sent = 0;
            Outgoing o;
            while (sent < 64 && outgoing.TryTake(out o, 0))
            {
                pubs[o.socket].SendFrame(o.data);
                sentCount[o.socket]++;
                sent++;
            }

            // 5. end of run: five copies 100 ms apart (one lost copy cannot hide it), then a short pause so they get out
            if (endRequested && endSent < 5 && now - lastEnd >= 100)
            {
                lastEnd = now;
                pubs[SSession].SendFrame(Protocol.BuildEndOfRun(endSeq++, endSimTime));
                endSent++;
                if (endSent == 5)
                {
                    Thread.Sleep(300);
                    return;
                }
            }

            if (sent == 0) Thread.Sleep(1);
        }
    }

    void HandleCommandMessage(byte[] raw)
    {
        Protocol.CmdType type;
        VehicleCommand c;
        string json, error;
        if (!Protocol.TryParseCommandMessage(raw, out type, out c, out json, out error))
        {
            if (Interlocked.Increment(ref parseErrors) <= 5) UnityEngine.Debug.LogError("NetworkHost: bad command message: " + error);
            return;
        }
        switch (type)
        {
            case Protocol.CmdType.Ready:
                readyJson = json;
                readyReceived = true;
                break;
            case Protocol.CmdType.Command:
                Interlocked.Increment(ref commandsReceived);
                if (!ReferenceEquals(executorRef, null)) executorRef.Submit(c);   // Submit is safe from any thread
                break;
            case Protocol.CmdType.Telemetry:
                latestTelemetry = json;
                break;
        }
    }

    void OnDestroy()
    {
        Time.timeScale = 1f;
        if (thread == null) return;

        if (started && !endRequested) RequestEndOfRun(Time.fixedTimeAsDouble);
        if (started) thread.Join(1500);        // five END_OF_RUN copies take about 0.5 s
        threadAlive = false;
        thread.Join(1000);
        NetMQConfig.Cleanup(false);

        string s = "NetworkHost: closed. sent";
        for (int i = 0; i < 5; i++) if (sentCount[i] > 0) s += " " + Names[i] + " " + sentCount[i];
        UnityEngine.Debug.Log(s + " | commands received " + commandsReceived + ", bad command messages " + parseErrors + ", queue drops " + queueDrops);
    }
}