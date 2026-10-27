from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from server.communication import protocol as P
from server.geometry.calibration import Calibration
from server.perception.building import BuildingConfig, BuildingFinder
from server.perception.ground import GroundConfig, GroundRemover
from server.perception.range_image import RangeImage, RangeImageBuilder
from server.recording.run_log import RunLog

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "processor"


@dataclass(frozen=True)
class MessageConfig:
    min_brake: float      # m/s^2, a new risk becomes a message only when the braking it needs reaches this


@dataclass(frozen=True)
class DebugConfig:
    keep_every: int       # every this many scans the ground labels are kept for the report (logging only)
    keep_max: int


@dataclass(frozen=True)
class ProcessorConfig:
    name: str
    ground: GroundConfig
    building: BuildingConfig
    messages: MessageConfig
    debug: DebugConfig


def _snake(key: str) -> str:
    return re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), key)


def _build(cls, d: dict, label: str):
    names = {f.name for f in fields(cls)}
    given = {_snake(k): v for k, v in d.items()}
    unknown = sorted(set(given) - names)
    missing = sorted(names - set(given))
    if unknown or missing:
        raise ValueError(f"processor config '{label}': unknown keys {unknown}, missing keys {missing}")
    return cls(**given)


def load_config(name: str) -> ProcessorConfig:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Available: {sorted(p.stem for p in CONFIG_DIR.glob('*.json'))}")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema") != 1 or d.get("name") != name:
        raise ValueError(f"{path}: schema must be 1 and name must be '{name}'")
    return ProcessorConfig(name, _build(GroundConfig, d["ground"], "ground"), _build(BuildingConfig, d["building"], "building"), _build(MessageConfig, d["messages"], "messages"),
                           _build(DebugConfig, d["debug"], "debug"))


@dataclass
class Frame:
    """What travels through the pipeline: every stage reads what the earlier ones filled in and fills in its own part."""
    header: P.Header
    bundle: P.Bundle
    image: RangeImage | None = None            # range image stage
    ground: np.ndarray | None = None           # ground stage: label per cell (NONE, GROUND, OTHER), the building stage changes some OTHER to BUILDING
    risks: dict[int, float] = field(default_factory=dict)    # object id -> stop position (odometer reading) seen in this frame
    removed: set[int] = field(default_factory=set)           # object ids whose message the processor drops now
    message: P.StopMessage | None = None       # message stage: the one message sent to the car


class RangeImageStage:
    name = "range_image"

    def __init__(self, calibration: Calibration):
        self.builder = RangeImageBuilder(calibration.lidar)

    def process(self, frame: Frame) -> None:
        if frame.bundle.lidar is not None:
            frame.image = self.builder.convert(frame.header, frame.bundle.lidar)


class GroundStage:
    name = "ground"

    def __init__(self, calibration: Calibration, calibration_json: dict, config: ProcessorConfig):
        self.remover = GroundRemover(RangeImageBuilder(calibration.lidar).geometry, config.ground)
        self.calibration = calibration
        self.calibration_json = calibration_json
        self.debug = config.debug
        self.n = 0
        self.kept_t, self.kept_ranges, self.kept_labels, self.kept_reasons, self.kept_camera = [], [], [], [], []
        self.camera_format = None        # (width, height, format) of the kept camera frames

    def process(self, frame: Frame) -> None:
        if frame.image is None:
            return
        frame.ground = self.remover.classify(frame.image)
        if self.n % self.debug.keep_every == 0 and len(self.kept_t) < self.debug.keep_max:
            self.kept_t.append(frame.header.t)
            self.kept_ranges.append(np.array(frame.bundle.lidar))
            self.kept_labels.append(frame.ground)
            self.kept_reasons.append(self.remover.reasons)
            cam = frame.bundle.camera
            self.kept_camera.append(np.frombuffer(cam.data, dtype=np.uint8).copy() if cam is not None else np.zeros(0, np.uint8))
            if cam is not None:
                self.camera_format = (cam.width, cam.height, int(cam.format))
        self.n += 1

    def save(self, run_dir: Path) -> None:
        if not self.kept_t:
            return
        lc = self.calibration.lidar
        sizes = np.array([len(c) for c in self.kept_camera])
        np.savez_compressed(run_dir / "ground_frames.npz", t=np.array(self.kept_t), ranges=np.stack(self.kept_ranges),
                            labels=np.stack(self.kept_labels), reasons=np.stack(self.kept_reasons), elevations_deg=lc.elevations_deg, azimuths_deg=lc.azimuths_deg,
                            mount=np.array([lc.mount.x, lc.mount.y, lc.mount.z]), camera_data=np.concatenate(self.kept_camera),
                            camera_offsets=np.concatenate([[0], np.cumsum(sizes)]), camera_format=np.array(self.camera_format or (0, 0, 0)),
                            calibration=np.array(json.dumps(self.calibration_json)))
        print(f"ground: {self.n} scans, {len(self.kept_t)} kept. Report: python -m server.ground_report {run_dir.parent.as_posix()}")


class BuildingStage:
    name = "building"

    def __init__(self, calibration: Calibration, config: ProcessorConfig):
        self.finder = BuildingFinder(RangeImageBuilder(calibration.lidar).geometry, config.building)
        self.kept_t, self.kept_links, self.kept_surfaces = [], [], []        # of every scan, for the report

    def process(self, frame: Frame) -> None:
        if frame.ground is None:
            return
        # changes the array in place: the ground stage keeps the same array for the report, so the report shows the buildings too
        self.finder.mark(frame.image, frame.ground)
        self.kept_t.append(frame.header.t)
        self.kept_links.append(self.finder.links)
        self.kept_surfaces.append(self.finder.surfaces)

    def save(self, run_dir: Path) -> None:
        if self.kept_t:
            np.savez_compressed(run_dir / "building_surfaces.npz", t=np.array(self.kept_t),
                                link_counts=np.array([len(k) for k in self.kept_links]), links=np.concatenate(self.kept_links),
                                surface_counts=np.array([len(k) for k in self.kept_surfaces]), surfaces=np.concatenate(self.kept_surfaces))


class MessageBook:
    """The processor's list of active messages, one per object id. An update only moves a stop position closer."""

    def __init__(self, min_brake: float):
        self.min_brake = min_brake
        self.stops: dict[int, float] = {}

    def offer(self, object_id: int, stop: float, speed: float, odometer: float) -> None:
        """A risk seen in this frame. A new one becomes a message only when the braking it needs reaches min_brake."""
        if object_id in self.stops:
            self.stops[object_id] = min(self.stops[object_id], stop)
            return
        remaining = stop - odometer
        if remaining <= 0.0 or speed * speed / (2.0 * remaining) >= self.min_brake:
            self.stops[object_id] = stop

    def remove(self, object_id: int) -> None:
        self.stops.pop(object_id, None)

    def nearest(self) -> P.StopMessage | None:
        if not self.stops:
            return None
        oid = min(self.stops, key=self.stops.get)
        return P.StopMessage(oid, self.stops[oid])


class MessageStage:
    name = "message"

    def __init__(self, config: ProcessorConfig):
        self.book = MessageBook(config.messages.min_brake)

    def process(self, frame: Frame) -> None:
        for oid in frame.removed:
            self.book.remove(oid)
        for oid, stop in frame.risks.items():
            self.book.offer(oid, stop, frame.header.speed, frame.bundle.odometer)
        frame.message = self.book.nearest()


class Pipeline:
    """The stages in order. Each gets the same Frame; the time of every stage goes to python/stages.csv."""

    def __init__(self, stages: list, run_log: RunLog):
        self.stages = stages
        self.run_dir = run_log.dir
        self.table = run_log.table("stages", ["t_ms"] + [s.name for s in stages])

    def run(self, frame: Frame) -> P.StopMessage | None:
        ms = []
        for s in self.stages:
            start = time.perf_counter_ns()
            s.process(frame)
            ms.append(f"{(time.perf_counter_ns() - start) / 1e6:.2f}")
        self.table.row(round(frame.header.t * 1000), *ms)
        return frame.message

    def save(self) -> None:
        for s in self.stages:
            if hasattr(s, "save"):
                s.save(self.run_dir)