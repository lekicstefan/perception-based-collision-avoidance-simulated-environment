"""What does a ZeroMQ PUB/SUB pair drop when the consumer stalls?  (step 4.3 experiment, pyzmq = libzmq)

A publisher sends numbered 38 kB messages (a LiDAR scan) every 10 ms with a small high-water mark (HWM).
The subscriber does not read for STALL seconds, then reads everything that is waiting.
We print which sequence numbers were delivered: the OLDEST ones (drop-new) or the NEWEST ones (drop-old).
"""
import struct, sys, threading, time
import zmq

N, PERIOD, STALL, SIZE = 300, 0.010, 1.0, 38492

def run(hwm, bufsize, port):
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB); sub = ctx.socket(zmq.SUB)
    for s in (pub, sub):
        s.setsockopt(zmq.LINGER, 0)
    pub.setsockopt(zmq.SNDHWM, hwm); sub.setsockopt(zmq.RCVHWM, hwm)
    if bufsize:
        pub.setsockopt(zmq.SNDBUF, bufsize); sub.setsockopt(zmq.RCVBUF, bufsize)
    pub.bind(f"tcp://127.0.0.1:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b""); sub.connect(f"tcp://127.0.0.1:{port}")
    time.sleep(0.5)                                     # let the subscription settle (slow joiner)
    pad = b"\x00" * (SIZE - 4)
    for i in range(N):                                  # publish while the subscriber is stalled
        pub.send(struct.pack("<I", i) + pad)
        time.sleep(PERIOD)
    time.sleep(STALL - N * PERIOD if STALL > N * PERIOD else 0)
    got = []
    idle_since = time.perf_counter()                    # keep reading until nothing has arrived for 0.5 s
    while time.perf_counter() - idle_since < 0.5:
        try:
            got.append(struct.unpack_from("<I", sub.recv(zmq.NOBLOCK))[0]); idle_since = time.perf_counter()
        except zmq.Again:
            time.sleep(0.005)
    pub.close(); sub.close(); ctx.term()
    return got

def describe(got):
    if not got: return "nothing received"
    runs, start = [], got[0]
    for a, b in zip(got, got[1:] + [None]):
        if b is None or b != a + 1:
            runs.append((start, a)); start = b
    return f"{len(got)} of {N} delivered; sequence runs {runs[:6]}"

if __name__ == "__main__":
    port = 27100
    for hwm, buf in ((5, 0), (5, 16384), (5, 1)):
        label = "default kernel buffers" if not buf else f"SNDBUF=RCVBUF={buf}"
        got = run(hwm, buf, port); port += 1
        print(f"HWM={hwm}, {label}: {describe(got)}")