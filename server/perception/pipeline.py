"""The perception pipeline of one LiDAR scan: ground removal, segmentation, cluster geometry, optional camera refinement."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from server.geometry.calibration import Calibration
from server.perception.camera_refinement import CameraBuffer, CameraRefiner, RefinementConfig
from server.perception.cluster_geometry import ClusterGeometry, GeometryExtractor
from server.perception.ground import GroundConfig, GroundRemover
from server.perception.range_image import RangeImage, RangeImageBuilder
from server.perception.segmentation import Segmentation, SegmentationConfig, Segmenter


@dataclass(frozen=True)
class PerceptionConfig:
    ground: GroundConfig = field(default_factory=GroundConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    size_prior_m: float = 1.0


@dataclass(frozen=True, eq=False)
class PerceptionResult:
    image: RangeImage
    ground: np.ndarray                  # bool (rows, cols)
    candidates: np.ndarray              # valid and not ground
    segmentation: Segmentation
    clusters: list                      # ClusterGeometry, with the camera refinement applied when it was used
    refinements: list                   # Refinement per cluster, empty when the refinement is off or has no frame


class PerceptionPipeline:
    def __init__(self, calibration: Calibration, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()
        self.builder = RangeImageBuilder(calibration.lidar)
        geo = self.builder.geometry
        self.ground = GroundRemover(geo, self.config.ground)
        self.segmenter = Segmenter(geo, self.config.segmentation)
        self.extractor = GeometryExtractor(geo, self.config.size_prior_m)
        self.refiner = CameraRefiner(calibration.camera, geo, self.config.refinement)
        self.camera_buffer = CameraBuffer()

    def process(self, img: RangeImage, cam_header=None, rgb: np.ndarray | None = None) -> PerceptionResult:
        ground = self.ground.ground_mask(img)
        candidates = img.valid & ~ground
        seg = self.segmenter.segment(img, candidates)
        geoms = self.extractor.extract_all(img, seg)
        geoms, refs = self.refiner.refine_and_apply(geoms, img, cam_header, rgb)
        return PerceptionResult(img, ground, candidates, seg, geoms, refs)

    def process_message(self, raw_lidar_message: bytes) -> PerceptionResult:
        """From a LIDAR message: takes the camera frame closest to the scan time from the camera buffer, if any."""
        img = self.builder.ingest(raw_lidar_message)
        cam = None
        if self.config.refinement.enabled:
            cam = self.camera_buffer.nearest(img.t, self.config.refinement.max_time_gap_s)
        return self.process(img, *(cam if cam else (None, None)))