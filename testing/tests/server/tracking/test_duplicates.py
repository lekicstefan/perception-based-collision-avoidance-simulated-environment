"""Two tracks of one object are merged although the cluster goes alternately to the one and to the other (step 6.11)."""
import copy
from dataclasses import replace

from server.tracking.track import TrackConfig
from server.tracking.tracker import Tracker, TrackerConfig
from testing.tests.server.tracking.test_extent import seen
from testing.tests.server.tracking.test_tracker import DT

PERSON = ([20.0, -0.25], [20.1, 0.25])
NEIGHBOUR = ([20.0, 0.75], [20.1, 1.25])                       # a second object, one metre across


def _with_duplicate(cfg):
    tr = Tracker(cfg)
    for k in range(8):
        tr.step([seen(*PERSON, k * DT)], k * DT)
    dup = copy.deepcopy(tr.tracks[0])
    dup.id = 99
    tr.tracks.append(dup)
    return tr


def test_a_duplicate_track_of_the_same_object_is_merged():
    cfg = TrackerConfig(track=TrackConfig(merge_scans=99))            # the merge test of the consecutive scans is out of the way
    tr = _with_duplicate(cfg)
    merged = []
    for k in range(8, 14):
        merged += tr.step([seen(*PERSON, k * DT)], k * DT).merged
    assert [t.id for t in tr.tracks] == [1] and merged == [(99, 1)]


def test_without_the_check_the_duplicate_stays_for_a_while():
    cfg = TrackerConfig(track=TrackConfig(merge_scans=99, duplicate_scans=0))
    tr = _with_duplicate(cfg)
    for k in range(8, 11):
        tr.step([seen(*PERSON, k * DT)], k * DT)
    assert len(tr.tracks) == 2


def test_two_objects_a_metre_apart_stay_two_tracks():
    tr = Tracker()
    for k in range(40):
        tr.step([seen(*PERSON, k * DT), seen(*NEIGHBOUR, k * DT)], k * DT)
    assert len(tr.tracks) == 2 and all(t.hits > 30 for t in tr.tracks)