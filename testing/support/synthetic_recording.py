"""Writes a small development-recording folder from synthetic scenes, in the format of the simulator's DevRecorder.

For tests of the evaluation and of the tools that read recordings, so they run without the Unity recording. The ego vehicle
stands still at the origin. Hazards (ids as in hazard_info.csv):
  1 car      static, 4.5 x 1.8 x 1.5 m, centre (25, -2.5)
  2 person   walks across 40 m ahead at 2 m/s, 0.6 x 0.5 x 1.7 m
  3 person   static far away at 78 m (only 1 to 2 cells hit it: not a detectable target)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from server.perception.range_image import RangeImageBuilder
from testing.support.dev_recording import lidar_dtype
from testing.support.synthetic_camera import make_camera, render as render_camera
from testing.support.synthetic_scene import Scene, make_lidar

HAZARDS = {1: ("car", "parked_car", 4.5, 1.8, 1.5), 2: ("person", "walker", 0.6, 0.5, 1.7), 3: ("person", "far_person", 0.6, 0.5, 1.7)}


def hazard_position(hid: int, t: float):
    if hid == 1:
        return 25.0, -2.5
    if hid == 2:
        return 40.0, -3.0 + 2.0 * t
    return 78.0, 0.0


def calibration_dict(lidar, camera) -> dict:
    mount = lambda m: {"x": m.x, "y": m.y, "z": m.z, "yawDeg": m.yaw_deg, "pitchDeg": m.pitch_deg, "rollDeg": m.roll_deg}
    return {"schema": 1, "calibrationId": "synthetic",
            "ego": {"wheelbase": 2.7, "length": 4.4, "width": 1.8, "height": 1.5, "rearOverhang": 0.9, "frontOverhang": 0.8},
            "camera": {"width": camera.width, "height": camera.height, "rateHz": 30.0, "format": "jpeg", "fx": camera.fx,
                       "fy": camera.fy, "cx": camera.cx, "cy": camera.cy, "horizontalFovDeg": camera.horizontal_fov_deg,
                       "verticalFovDeg": camera.vertical_fov_deg, "mount": mount(camera.mount),
                       "opticalFromVehicle": {"R": camera.r_optical_from_vehicle.ravel().tolist(),
                                              "t": camera.t_optical_from_vehicle.tolist()}},
            "lidar": {"rows": lidar.rows, "cols": lidar.cols, "rateHz": 10.0, "maxRangeM": lidar.max_range_m,
                      "mount": mount(lidar.mount), "elevationsDeg": lidar.elevations_deg.tolist(),
                      "azimuthsDeg": lidar.azimuths_deg.tolist()},
            "poseRateHz": 50.0}


def write_recording(folder, scans: int = 12, seed: int = 0) -> Path:
    folder = Path(folder)
    (folder / "camera").mkdir(parents=True, exist_ok=True)
    lidar, camera = make_lidar(cols=601), make_camera()
    (folder / "calibration.json").write_text(json.dumps(calibration_dict(lidar, camera)), encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps({"formatVersion": 1, "scene": "synthetic", "sensorConfig": "synthetic",
                                                  "seed": seed, "rows": lidar.rows, "cols": lidar.cols, "lidarRateHz": 10.0}))
    builder = RangeImageBuilder(lidar)

    def scene_at(t):
        sc = Scene()
        for hid in HAZARDS:
            x, y = hazard_position(hid, t)
            _, _, length, width, height = HAZARDS[hid]
            sc.add_box(x, y, length, width, height)
        return sc

    dtype = lidar_dtype(lidar.rows, lidar.cols)
    records = np.zeros(scans, dtype=dtype)
    vis_rows = []
    for k in range(scans):
        t = round(0.1 * k, 6)
        sc = scene_at(t)
        img, truth, _ = sc.render(lidar)
        raw = np.where(img.valid, np.nan_to_num(img.range_m) * 100, 0).round().astype(np.uint16)
        raw[img.state == 2] = 65535
        records[k] = (k, t, (0, 0, 0, 0, 0, 0), (0, 0, 0), raw)
        for i, hid in enumerate(HAZARDS):
            hits = int((sc.last_box_index == i).sum())
            x, y = hazard_position(hid, t)
            vis_rows.append(f"{t},{k},{hid},{hits},{hits},{max(0.0, x - 2):.3f},{x:.3f},{np.degrees(np.arctan2(y, x)):.3f},{int(hits > 0)}")
    records.tofile(folder / "lidar.bin")
    (folder / "visibility.csv").write_text("t,scan,id,hits_geometric,hits_returned,min_range_m,centre_range_m,azimuth_deg,in_view\n"
                                           + "\n".join(vis_rows) + "\n")

    duration = 0.1 * scans
    cam_rows = []
    for f, t in enumerate(np.arange(0.0, duration, 1 / 30)):
        boxes = [(*hazard_position(h, t), *HAZARDS[h][2:]) for h in HAZARDS]
        cv2.imwrite(str(folder / "camera" / f"{f:06d}.jpg"), cv2.cvtColor(render_camera(camera, boxes, seed=f), cv2.COLOR_RGB2BGR))
        cam_rows.append(f"{f},{t:.6f},0,0,0,0,0,0,0,0,0,0")
    (folder / "camera.csv").write_text("frame,t,x,y,z,yaw,pitch,roll,speed,yaw_rate,steering,bytes\n" + "\n".join(cam_rows) + "\n")
    (folder / "pose.csv").write_text("t,x,y,z,yaw,pitch,roll,speed,yaw_rate,steering\n"
                                     + "\n".join(f"{t:.4f},0,0,0,0,0,0,0,0,0" for t in np.arange(0.0, duration, 0.02)) + "\n")

    steps = np.arange(0.0, duration, 0.01)
    (folder / "ego.csv").write_text("t,x,y,z,yaw_rad,pitch_rad,speed,yaw_rate,steering,accel,jerk,requested_accel\n"
                                    + "\n".join(f"{t:.4f},0,0,0,0,0,0,0,0,0,0,0" for t in steps) + "\n")
    haz = []
    for t in steps:
        for hid, (_, _, length, width, height) in HAZARDS.items():
            x, y = hazard_position(hid, t)
            vy = 2.0 if hid == 2 else 0.0
            haz.append(f"{t:.4f},{hid},{x:.4f},{y:.4f},{height / 2:.4f},0,0,{vy},{abs(vy)},{length},{width},{height}")
    (folder / "hazards.csv").write_text("t,id,x,y,z,yaw_rad,vx,vy,speed,length,width,height\n" + "\n".join(haz) + "\n")
    (folder / "hazard_info.csv").write_text("id,name,label,length,width,height,colliders\n" + "\n".join(
        f"{hid},{n},{lab},{l},{w},{h},1" for hid, (lab, n, l, w, h) in HAZARDS.items()) + "\n")
    return folder