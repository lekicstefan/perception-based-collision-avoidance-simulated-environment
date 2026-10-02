import struct
import time
import zmq

SENSOR = "tcp://127.0.0.1:5555"
COMMAND = "tcp://127.0.0.1:5556"
MSG_HELLO, MSG_STATE = 1, 2
CMD_READY, CMD_ECHO = 1, 3
STATE_FMT = "<Bdiff"

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(SENSOR)
cmd = ctx.socket(zmq.PUB)
cmd.bind(COMMAND)

seen_hello = False
running = False
last_ready = 0.0
count = 0

print("waiting for Unity...", flush=True)
try:
    while True:
        if sub.poll(20):
            mtype, sim_t, fid, z, speed = struct.unpack(STATE_FMT, sub.recv())
            if mtype == MSG_HELLO:
                if not seen_hello:
                    print("HELLO received, sending READY", flush=True)
                seen_hello = True
            elif mtype == MSG_STATE:
                running = True
                cmd.send(struct.pack("<BI", CMD_ECHO, fid))
                count += 1
                if count % 500 == 0:
                    print(f"echoed {count} frames (last id {fid})", flush=True)

        if seen_hello and not running and time.perf_counter() - last_ready >= 0.1:
            cmd.send(bytes([CMD_READY]))
            last_ready = time.perf_counter()
except KeyboardInterrupt:
    print(f"done, echoed {count} frames")