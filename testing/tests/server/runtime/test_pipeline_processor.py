"""The processor around the stage (step 7.13): messages in, one command per scan out. Perception is replaced by a stub that returns the
ray-cast image, so this tests the wiring (order, one command per cycle, no scan no command), not the perception."""
from types import SimpleNamespace

from server.communication import protocol as P
from server.communication.transport import Incoming
from server.perception.ground import GroundConfig, GroundRemover
from server.risk.ego_path import EgoGeometry
from server.risk.stage import RiskStage
from server.runtime.pipeline_processor import PipelineProcessor
from server.tracking.tracker import Tracker
from testing.support.synthetic_scene import Scene, make_lidar

CAL = {"calibrationId": "stub"}


def msg(mtype, t, seq=0):
    return Incoming(P.parse_header(P.build_message(mtype, seq, t=t, speed=10.0)), b"")


def processor(times):
    img0, _, geo = Scene().render(make_lidar())
    ground = GroundRemover(geo, GroundConfig()).ground_mask(img0)
    pending = list(times)

    def process_message(raw):
        img = type(img0)(pending.pop(0), img0.seq, img0.pose, 10.0, 0.0, 0.0, img0.range_m, img0.state)
        return SimpleNamespace(image=img, ground=ground, clusters=[])

    p = PipelineProcessor()
    p._calibration_dict = CAL                                                       # skips the build from a real calibration
    p.pipeline = SimpleNamespace(process_message=process_message, builder=SimpleNamespace(geometry=geo))
    p.tracker = Tracker()
    p.stage = RiskStage(EgoGeometry(), geo, p.config.stage)
    return p


def test_a_scan_gives_one_command_with_its_time():
    p = processor([0.5])
    c = p.step([msg(P.MsgType.POSE, 0.49), msg(P.MsgType.LIDAR, 0.5)], CAL)
    assert c.id == 1 and c.data_timestamp == 0.5 and c.safety_cap == 40.0


def test_without_a_scan_there_is_no_command():
    p = processor([])
    assert p.step([msg(P.MsgType.POSE, 0.49)], CAL) is None
    assert p.step([], CAL) is None


def test_two_scans_in_a_cycle_both_run_and_the_last_command_is_returned():
    p = processor([0.5, 0.6])
    c = p.step([msg(P.MsgType.LIDAR, 0.5), msg(P.MsgType.LIDAR, 0.6, seq=1)], CAL)
    assert c.data_timestamp == 0.6 and p.stage.next_id == 3