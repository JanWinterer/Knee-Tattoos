"""Marker trackers: one file per marker system, all with the same interface.

    from marker_tracking.trackers import get_tracker, list_trackers
    list_trackers()                                     # ['aruco_4x4_50', 'aruco_5x5_50', ...]
    tracker = get_tracker('aruco_4x4_50')               # default settings
    tracker = get_tracker('aruco_4x4_50', refinement='none')

The interface (what a tracker gets, what it must return) is described in base.py.

Adding a tracker:
    1. copy _template.py to <name>.py in this folder (no leading underscore)
    2. fill in detect() and register the class under a new name
That is all. Every module in this folder except _*.py is imported below, which
runs its register() call; run.py, the CSVs and the tests find it by name.
"""
import importlib
import pkgutil

from marker_tracking.trackers.base import REGISTRY, Detection, Tracker, register  # noqa: F401 (re-exported)

# Tracker modules whose third-party package is not installed: {module name: error}.
# Their trackers are missing from list_trackers(); get_tracker() reports why.
IMPORT_ERRORS = {}


def _import_tracker_modules():
    """Import every module in this folder (except _*.py) so their register() calls run."""
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith('_'):
            continue  # templates and private helpers
        try:
            importlib.import_module(f'{__name__}.{module.name}')
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith('marker_tracking'):
                raise  # a bug in our own code, not a missing optional package
            IMPORT_ERRORS[module.name] = e


def list_trackers():
    """Names of all available trackers, sorted."""
    return sorted(REGISTRY)


def get_tracker(name, **settings):
    """Create a new tracker by name. `settings` are passed to its constructor."""
    if name not in REGISTRY:
        message = f'unknown tracker {name!r}; available: {list_trackers()}'
        if IMPORT_ERRORS:
            message += f'; tracker modules that could not be imported: {IMPORT_ERRORS}'
        raise ValueError(message)
    tracker = REGISTRY[name](**settings)
    tracker.name = name
    return tracker


_import_tracker_modules()
