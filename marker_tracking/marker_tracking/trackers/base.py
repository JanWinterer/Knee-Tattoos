"""The tracker contract: what every marker tracker gets and what it must return.

A tracker finds the markers of ONE marker system in a colour image and returns
2D positions only. Depth, 3D, the global frame and the CSVs are shared code
outside trackers/, so every tracker is measured in exactly the same way.

Per take, detect.py calls
    tracker.start(camera, markers)       once, before the first frame
    tracker.detect(color, depth_m)       once per frame, in frame order
and checks every returned Detection:
    - marker_id   int >= 0
    - u_px, v_px  marker centre in pixels, finite and inside the image;
                  (0, 0) = centre of the top-left pixel (same as io.py)
    - extra       optional tracker-specific numbers (corners, own pose estimate, scores, ...);
                  every key must be listed in the tracker's extra_columns.
                  Extras go to markers_extra.csv and never replace the depth-based xyz.
    - the same id twice in one frame is ambiguous: all copies are dropped.

New tracker: copy _template.py (see trackers/__init__.py).
"""
from dataclasses import dataclass, field

# Tracker name -> factory(**settings) that returns a new Tracker. Filled by register().
REGISTRY = {}


@dataclass
class Detection:
    """One marker found in one frame."""
    marker_id: int
    u_px: float                                  # marker centre, pixels
    v_px: float
    extra: dict = field(default_factory=dict)   # {column name: number}, keys from Tracker.extra_columns


class Tracker:
    """Base class of all trackers. Subclass it and register the subclass (see register())."""

    name = None          # the registered name, set by trackers.get_tracker()
    extra_columns = []   # names of the values a Detection may carry in .extra, in CSV column order
    camera = None        # io.Camera of the current take, set by start()
    markers = None       # markers.json of the current take ({'tracker', 'ids', 'size_m'}) or None

    def start(self, camera, markers):
        """Called once per take, before the first frame.

        Subclasses that keep state from frame to frame reset it here
        (and call super().start(camera, markers) first).
        """
        self.camera = camera
        self.markers = markers

    def detect(self, color, depth_m):
        """Find markers in one frame and return a list of Detection.

        color:   BGR uint8 (H, W, 3)
        depth_m: float32 (H, W), metres, 0 = no depth. Available for trackers
                 that need it; the 3D position is still computed outside the tracker.
        """
        raise NotImplementedError

    def settings(self):
        """Every parameter that changes the result, as a JSON-able dict (stored in run.json)."""
        return {}

    def max_id(self):
        """Largest marker id this system can produce, or None if there is no fixed limit."""
        return None

    def render(self, marker_id, size_px):
        """Optional: grey uint8 (size_px, size_px) image of one marker, for printing and synthetic takes.

        Trackers that implement it are tested automatically by tests/test_trackers.py.
        """
        raise NotImplementedError(f'tracker {self.name!r} cannot render markers')


def register(name, factory=None):
    """Make a tracker available under `name`.

    As a class decorator (one class, one tracker):
        @register('my_tracker')
        class MyTracker(Tracker): ...

    With a factory (one class, several configurations):
        register('aruco_4x4_50', functools.partial(ArucoTracker, 'DICT_4X4_50'))

    The factory is called as factory(**settings) by trackers.get_tracker().
    """
    if factory is None:
        # Used as a decorator: register the class itself and hand it back unchanged.
        def decorator(cls):
            register(name, cls)
            return cls
        return decorator
    if name in REGISTRY:
        raise ValueError(f'tracker name {name!r} is registered twice')
    REGISTRY[name] = factory
    return factory
