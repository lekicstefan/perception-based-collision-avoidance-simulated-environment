import struct
import time
import numpy as np
import zmq

SENSOR = "tcp://127.0.0.1:5555"
COMMAND = "tcp://127.0.0.1:5556"
MSG_HELLO, MSG_STATE = 1, 2
CMD_READY = 1
FMT = "<Bdi"  # type, sim time (float64), frame id (int32) = 13 bytes

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(SENSOR)
cmd = ctx.socket(zmq.PUB)
cmd.bind(COMMAND)

seen_hello = False
running = False
last_ready = 0.0
prev_fid = None
lost = 0
wall, sim = [], []

print("waiting for Unity...", flush=True)
try:
    while True:
        if sub.poll(20):
            data = sub.recv()
            now = time.perf_counter()
            mtype, sim_t, fid = struct.unpack(FMT, data)

            if mtype == MSG_HELLO:
                if not seen_hello:
                    print("HELLO received, sending READY", flush=True)
                seen_hello = True

            elif mtype == MSG_STATE:
                if not running:
                    running = True

                    if fid == 0:
                        print("first STATE frame id = 0: no messages lost at start", flush=True)
                    else:
                        print(f"WARNING: first STATE frame id = {fid}, start was lost", flush=True)

                if prev_fid is not None and fid != prev_fid + 1:
                    gap = fid - prev_fid - 1
                    lost += gap

                    print(f"WARNING: {gap} message(s) missing before frame {fid}", flush=True)

                prev_fid = fid

                wall.append(now)
                sim.append(sim_t)

                print(f"frame {fid:5d}  sim_t={sim_t:8.3f}", flush=True)

        if seen_hello and not running and time.perf_counter() - last_ready >= 0.1:
            cmd.send(bytes([CMD_READY]))
            last_ready = time.perf_counter()

except KeyboardInterrupt:
    pass

if len(wall) > 2:
    wall = np.array(wall)
    sim = np.array(sim)
    inter = np.diff(wall) * 1000
    offset = wall - sim
    rel = (offset - offset.min()) * 1000

    print(f"\nmessages: {len(wall)}, lost: {lost}")
    print(f"inter-arrival ms: mean {inter.mean():.1f}, std {inter.std():.1f}, max {inter.max():.1f}")
    print(f"delay above best case ms: p50 {np.percentile(rel,50):.1f}, "f"p95 {np.percentile(rel,95):.1f}, p99 {np.percentile(rel,99):.1f}, max {rel.max():.1f}")
    print(f"delay trend first-vs-last 10%: {rel[:len(rel)//10].mean():.1f} -> {rel[-(len(rel)//10):].mean():.1f} ms")