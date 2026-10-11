"""The top view draws the centroid, the bounding box and the ground truth when it is given (step 6.3)."""
import numpy as np

from testing.support import scan_view
from testing.support.ground_truth import TruthObject
from testing.tests.support.test_scan_view import CALIB, result_for
from testing.support.synthetic_scene import Scene


def test_the_truth_footprint_is_drawn_in_green_and_only_when_given():
    result, geo = result_for(Scene().add_box(20, 0, 4.5, 1.8, 1.5))
    truth = [TruthObject(1, "car", np.array([20.0, 0.0]), 0.0, 4.5, 1.8, 1.5, None, True)]
    plain = scan_view.top_view(result, CALIB, geo)
    drawn = scan_view.top_view(result, CALIB, geo, truths=truth)
    green = lambda im: int(((im[..., 1] == 200) & (im[..., 0] == 70)).sum())  # noqa: E731
    assert green(plain) == 0 and green(drawn) > 0                      # thin anti-aliased lines: few pixels have the exact colour
    assert int((plain != drawn).any(axis=-1).sum()) > 100               # the outline and the label change the picture
    assert plain.shape == drawn.shape


def test_a_focus_window_is_a_square_around_the_point():
    result, geo = result_for(Scene().add_box(20, 0, 4.5, 1.8, 1.5))
    view = scan_view.top_view(result, CALIB, geo, focus=(20.0, 0.0, 6.0))
    assert view.shape == (500, 500, 3)
    assert int((view != scan_view.BACKGROUND).any(axis=-1).sum()) > 200       # the car is inside the window