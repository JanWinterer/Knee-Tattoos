"""Template for a new tracker. Not loaded (the leading underscore hides it).

1. Copy this file to trackers/<name>.py (no leading underscore).
2. Rename the class and the registered name, fill in detect().
3. Run it:  python -m marker_tracking.run --take <take> --out <folder> --tracker my_tracker

The rules a Detection must follow are in base.py. If you implement render(),
tests/test_trackers.py checks the new tracker automatically.
"""
from marker_tracking.trackers.base import Detection, Tracker, register


@register('my_tracker')
class MyTracker(Tracker):
    # Tracker-specific numbers a Detection may carry in .extra (written to markers_extra.csv).
    # Leave empty if there are none.
    extra_columns = ['score']

    def __init__(self, threshold=0.5):
        # Every setting that changes the result: a constructor argument with a default,
        # so it can be set via get_tracker('my_tracker', threshold=0.7) or run.py --set threshold=0.7.
        self.threshold = threshold

    def settings(self):
        return {'threshold': self.threshold}  # stored in run.json

    def start(self, camera, markers):
        super().start(camera, markers)  # sets self.camera and self.markers
        # Reset state that is carried from frame to frame here.

    def detect(self, color, depth_m):
        detections = []
        # Find the markers in `color` (BGR uint8) and append one Detection per marker, e.g.
        #   detections.append(Detection(marker_id=3, u_px=412.3, v_px=218.9, extra={'score': 0.93}))
        return detections
