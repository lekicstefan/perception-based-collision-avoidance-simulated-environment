import struct
import time
import zmq

SENSOR = "tcp://127.0.0.1:5555"
COMMAND = "tcp://127.0.0.1:5556"
MSG_HELLO, MSG_STATE = 1, 2
CMD_READY, CMD_BRAKE = 1, 2
STATE_FMT = "<Bdiff"   # type, sim_t, frame_id, z, speed = 21 bytes

OBSTACLE_NEAR_Z = 79.5     # TEMPORARY: read from the scene
CAR_HALF_LENGTH = 2.1      # TEMPORARY: half of the car's z scale
BRAKE_GAP = 20.0           # TEMPORARY shortcut: brake when gap <= 20 m
BRAKE_DECEL = 6.0          # m/s^2

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(SENSOR)
cmd = ctx.socket(zmq.PUB)
cmd.bind(COMMAND)

seen_hello = False
running = False
last_ready = 0.0
cmd_id = 0
braking = False
stopped = False

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
                gap = OBSTACLE_NEAR_Z - (z + CAR_HALF_LENGTH)
                if gap <= BRAKE_GAP:
                    cmd_id += 1
                    cmd.send(struct.pack("<BIf", CMD_BRAKE, cmd_id, BRAKE_DECEL))
                    if not braking:
                        braking = True
                        print(f"BRAKE sent at gap={gap:.2f} m, speed={speed:.2f} m/s", flush=True)
                if braking and speed < 0.01 and not stopped:
                    stopped = True
                    print(f"STOPPED, final gap = {gap:.2f} m", flush=True)
                print(f"frame {fid:5d} sim_t={sim_t:7.2f} gap={gap:6.2f} v={speed:5.2f}", flush=True)

        if seen_hello and not running and time.perf_counter() - last_ready >= 0.1:
            cmd.send(bytes([CMD_READY]))
            last_ready = time.perf_counter()
except KeyboardInterrupt:
    pass