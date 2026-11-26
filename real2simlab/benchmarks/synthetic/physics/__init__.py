"""Truth generators of the synthetic suite: exact or high-precision physics, NumPy and SciPy only.

Each module defines the phenomenon's objects, hidden parameters, recorded setup (``SETUP``),
camera, ``simulate`` and ``scene`` (for rendering). The package is copied verbatim into
generator twins, so it never imports Real2SimLab.
"""

from . import bouncing_ball, collision, incline_slider, pendulum, projectile

PHENOMENA = {
    module.NAME: module
    for module in (bouncing_ball, pendulum, incline_slider, projectile, collision)
}


def render_video_frames(module, tracks, setup):  # type: ignore[no-untyped-def]
    """(F, H, W, 3) uint8 frames of a simulated phenomenon with ``setup``."""
    from .render import render_frames

    count = len(next(iter(tracks.values()))["position"])
    return render_frames(
        module.CAMERA,
        module.static_scene(setup),
        [module.dynamic_scene(tracks, index) for index in range(count)],
    )
