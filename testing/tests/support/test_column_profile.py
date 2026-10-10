from server.perception.pipeline import PerceptionPipeline
from testing.support.synthetic_scene import Scene, make_lidar
from testing.tools.perception.column_profile import profile_text


def test_profile_of_a_person_flush_against_a_car_shows_the_height_cut():
    from server.geometry.calibration import Calibration, EgoCalibration
    from testing.support.synthetic_camera import make_camera
    lidar = make_lidar()
    calib = Calibration("t", EgoCalibration(2.7, 4.4, 1.8, 1.5, 0.9, 0.8), make_camera(), lidar, 100.0)
    pipe = PerceptionPipeline(calib)
    img, _, _ = Scene().add_box(12, 0, 4.5, 1.8, 1.5).add_box(10.0, 1.15, 0.6, 0.5, 1.8).render(lidar)
    text = profile_text(pipe, img, pipe.process(img))
    assert "HEIGHT" in text and text.startswith("cluster 0")