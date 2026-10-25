using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
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
    const int SSession = 0, SBundle = 1, SOracle = 2;
    static readonly int[] Hwm = { 1000, 3, 5 };
    static readonly int[] Ports = { Protocol.PortSession, Protocol.PortBundle, Protocol.PortOracle };
    static readonly string[] Names = { "session", "bundle", "oracle" };
    const int SocketBufferBytes = 64 * 1024;    // bounds how stale queued data can get (see the stall experiment)
    const int MaxQueued = 500;                  // safety net, the network thread normally keeps the queue near empty

    struct Outgoing { public int socket; public byte[] data; public int frame; }   // frame = capture time in ms for bundles (the latency key), -1 for everything else
    struct Pending { public LidarScan scan; public CameraFrame frame; public long lidarReady, cameraReady; }

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
    uint oracleSeq;                                 // main thread only
    readonly Dictionary<int, Pending> pending = new Dictionary<int, Pending>();   // main thread only
    bool needLidar, needCamera;
    int incompleteFrames;
    string latencyPath;
    readonly long[] sentCount = new long[3];        // network thread writes, read after Join
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
        if (oracle != null && oracle.oracleMode) oracle.OracleReady += OnOracle;
    }

    void OnDisable()
    {
        if (lidar != null) lidar.ScanReady -= OnScan;
        if (cameraSensor != null) cameraSensor.FrameReady -= OnFrame;
        if (oracle != null) oracle.OracleReady -= OnOracle;
    }

    void Start()
    {
        if (!waitForProcessor) return;
        if (calibration == null) UnityEngine.Debug.LogWarning("NetworkHost: no CalibrationProvider assigned, the processor will never get a calibration.");
        executorRef = executor;
        needLidar = lidar != null && lidar.rig.Config.lidar.enabled;
        needCamera = cameraSensor != null && cameraSensor.rig.Config.camera.enabled;
        Latency.Reset();
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
            if (RunFolder.Current == RunFolder.State.Enabled) latencyPath = System.IO.Path.Combine(RunFolder.UnityDir, "latency.csv");

            started = true;
            Time.timeScale = 1f;
            UnityEngine.Debug.Log("NetworkHost: READY received, the run starts now");
        }
    }

    // ------------------------------------------------------------------ main thread: sensor events to the queue

    // Lidar, camera and pose belong to one frame. The lidar scan and the pose are ready at the capture step; the camera
    // frame arrives later (readback and JPEG encoding), and the bundle leaves when the last part is there.
    void OnScan(LidarScan s)
    {
        if (!started) return;
        Pending p;
        pending.TryGetValue(s.frameId, out p);
        p.scan = s;
        p.lidarReady = Latency.Ns();
        Offer(s.frameId, p);
    }

    void OnFrame(CameraFrame f)
    {
        if (!started) return;
        Pending p;
        pending.TryGetValue(f.frameId, out p);
        p.frame = f;
        p.cameraReady = Latency.Ns();
        Offer(f.frameId, p);
    }

    void Offer(int id, Pending p)
    {
        if ((needLidar && p.scan == null) || (needCamera && p.frame == null))
        {
            pending[id] = p;
            return;
        }
        pending.Remove(id);
        long captureTicks = p.scan != null ? p.scan.captureTicks : p.frame.captureTicks;
        if (p.scan != null && p.frame != null) captureTicks = System.Math.Min(p.scan.captureTicks, p.frame.captureTicks);
        long captureNs = Latency.TicksToNs(captureTicks);
        long ready = System.Math.Max(p.lidarReady, p.cameraReady);
        byte[] msg = Protocol.BuildBundle(id, p.scan, p.frame, captureNs);
        long built = Latency.Ns();
        Protocol.Stamp(msg, Protocol.BundleBuiltOffset, built);
        double simTime = p.scan != null ? p.scan.timestamp : p.frame.timestamp;
        uint key = Latency.Key(simTime);
        Latency.Frame(key, simTime, captureNs, p.lidarReady, p.cameraReady, built, msg.Length);
        Enqueue(SBundle, msg, (int)key);

        // a part that never arrives (camera readback error, encoder queue full) must not keep a frame forever
        List<int> stale = null;
        foreach (int k in pending.Keys) if (k < id - 3) (stale ?? (stale = new List<int>())).Add(k);
        if (stale != null)
            foreach (int k in stale)
            {
                pending.Remove(k);
                if (++incompleteFrames == 1) UnityEngine.Debug.LogWarning("NetworkHost: a frame never became complete (lidar or camera part missing) and was dropped");
            }
    }

    void OnOracle(OracleFrame o) { if (started) Enqueue(SOracle, Protocol.BuildOracle(oracleSeq++, o), -1); }

    void Enqueue(int socket, byte[] data, int frame)
    {
        if (outgoing.Count >= MaxQueued)
        {
            if (Interlocked.Increment(ref queueDrops) == 1) UnityEngine.Debug.LogWarning("NetworkHost: send queue is full, dropping messages");
            return;
        }
        outgoing.Add(new Outgoing { socket = socket, data = data, frame = frame });
    }

    // ------------------------------------------------------------------ network thread

    void NetLoop(bool oracleMode)
    {
        var pubs = new PublisherSocket[3];
        SubscriberSocket cmd = null;
        try
        {
            AsyncIO.ForceDotNet.Force();       // needed for NetMQ under Mono in the player (found in phase 1)
            int n = oracleMode ? 3 : 2;        // the oracle socket does not exist in normal mode
            for (int i = 0; i < n; i++) pubs[i] = BindPublisher(i);
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

    // After a restart the previous run's sockets may still be closing: a busy port is retried for up to 10 s before it is an error.
    PublisherSocket BindPublisher(int i)
    {
        string address = "tcp://127.0.0.1:" + Ports[i];
        var waited = Stopwatch.StartNew();
        while (true)
        {
            var p = new PublisherSocket();
            p.Options.SendHighWatermark = Hwm[i];
            p.Options.Linger = TimeSpan.FromMilliseconds(500);
            if (i != SSession) p.Options.SendBuffer = SocketBufferBytes;
            try
            {
                p.Bind(address);
                return p;
            }
            catch (NetMQException e)
            {
                p.Dispose();
                if (waited.ElapsedMilliseconds > 10000 || !threadAlive) throw;
                UnityEngine.Debug.LogWarning("NetworkHost: " + Names[i] + " port is busy, retrying (the previous run is still closing): " + e.Message);
                Thread.Sleep(200);
            }
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
            while (cmd.TryReceiveFrameBytes(TimeSpan.Zero, out raw)) HandleCommandMessage(raw, Latency.Ns());

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
                long sentNs = 0;
                if (o.frame >= 0)
                {
                    sentNs = Latency.Ns();
                    Protocol.Stamp(o.data, Protocol.BundleSentOffset, sentNs);
                }
                pubs[o.socket].SendFrame(o.data);
                if (o.frame >= 0) Latency.Sent((uint)o.frame, sentNs);
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

    void HandleCommandMessage(byte[] raw, long receivedNs)
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
                Latency.CommandReceived(Latency.Key(c.dataTimestamp), receivedNs);
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

        if (started && !endRequested) RequestEndOfRun(SimClock.Now);
        if (started) thread.Join(1500);        // five END_OF_RUN copies take about 0.5 s
        threadAlive = false;
        thread.Join(1000);
        NetMQConfig.Cleanup(false);
        if (latencyPath != null) Latency.Write(latencyPath);

        string s = "NetworkHost: closed. sent";
        for (int i = 0; i < 3; i++) if (sentCount[i] > 0) s += " " + Names[i] + " " + sentCount[i];
        UnityEngine.Debug.Log(s + " | commands received " + commandsReceived + ", bad command messages " + parseErrors + ", queue drops " + queueDrops + ", incomplete frames " + incompleteFrames);
    }
}