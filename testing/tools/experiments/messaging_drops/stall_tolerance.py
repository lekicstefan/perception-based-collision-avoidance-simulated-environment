"""How long can the processor stall before each stream loses messages?  (step 4.3 experiment)

Three streams at the real rates and sizes (pose 50 Hz 88 B, camera 30 Hz ~30 kB, LiDAR 10 Hz 38 kB) on separate
PUB/SUB pairs with the HWMs of docs/protocol.md. The consumer runs a 33 ms cycle (drain everything, then sleep) and
freezes once for STALL seconds. We count lost messages per stream from the sequence numbers.
"""
import struct, threading, time
import zmq

STREAMS = {"pose": (50, 88, 100), "camera": (30, 30000, 10), "lidar": (10, 38492, 5)}   # Hz, bytes, HWM
DURATION = 9.0

def trial(stall, port0, bufsize=0):
    ctx = zmq.Context(); pubs, subs = {}, {}
    for i, (name, (hz, size, hwm)) in enumerate(STREAMS.items()):
        p = ctx.socket(zmq.PUB); s = ctx.socket(zmq.SUB)
        p.setsockopt(zmq.SNDHWM, hwm); s.setsockopt(zmq.RCVHWM, hwm)
        p.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.LINGER, 0)
        if bufsize: p.setsockopt(zmq.SNDBUF, bufsize); s.setsockopt(zmq.RCVBUF, bufsize)
        p.bind(f"tcp://127.0.0.1:{port0+i}"); s.setsockopt(zmq.SUBSCRIBE, b""); s.connect(f"tcp://127.0.0.1:{port0+i}")
        pubs[name], subs[name] = p, s
    time.sleep(0.5)
    stop = threading.Event()
    def publish(name):
        hz, size, _ = STREAMS[name]; pad = b"\x00" * (size - 4); i = 0; t0 = time.perf_counter()
        while not stop.is_set():
            pubs[name].send(struct.pack("<I", i) + pad); i += 1
            time.sleep(max(0, t0 + i / hz - time.perf_counter()))
    threads = [threading.Thread(target=publish, args=(n,)) for n in STREAMS]
    for t in threads: t.start()
    recv = {n: [] for n in STREAMS}; t0 = time.perf_counter(); stalled = False
    while time.perf_counter() - t0 < DURATION:
        if not stalled and time.perf_counter() - t0 > 2.0:
            stalled = True; time.sleep(stall)
        for n, s in subs.items():
            while True:
                try: recv[n].append(struct.unpack_from("<I", s.recv(zmq.NOBLOCK))[0])
                except zmq.Again: break
        time.sleep(0.033)
    stop.set(); [t.join() for t in threads]
    lost = {n: (max(r) + 1 - len(set(r))) if r else 0 for n, r in recv.items()}
    # backlog = how many messages were delivered in the first cycle after the stall (stale data the processor must chew through)
    for s in list(subs.values()) + list(pubs.values()): s.close()
    ctx.term()
    return lost

if __name__ == "__main__":
    port = 27300
    for buf, label in ((0, "default kernel buffers"), (65536, "SNDBUF = RCVBUF = 64 kB")):
        print(f"--- {label}")
        print("stall   lost pose   lost camera   lost lidar")
        for stall in (0.0, 0.5, 1.0, 2.0, 4.0):
            l = trial(stall, port, buf); port += 5
            print(f"{stall:4.1f} s  {l['pose']:9d}   {l['camera']:11d}   {l['lidar']:10d}")