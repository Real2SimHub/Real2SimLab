"""What a task lets a method see of a capture."""

from __future__ import annotations

from typing import Self

from pydantic import Field, NonNegativeInt, model_validator

from real2simlab._base import StrictModel
from real2simlab.capture import paths
from real2simlab.capture.model import Capture, FieldPath


class AuditRecord(StrictModel):
    """What a visibility policy withheld from one capture."""

    policy: str
    capture_id: str
    hidden: tuple[str, ...]
    frame_window: tuple[int, int] | None = None


class VisibilityPolicy(StrictModel):
    """A rule for the method-visible part of a capture.

    ``expose`` lists the field paths a method may see, on top of the always-visible ones.
    ``frame_window`` keeps only visible frames ``[start, end)`` of every view, e.g. when a
    benchmark shows the first seconds of a clip. It does not trim robot trajectories; hide
    ``robot`` if its later samples must stay unseen.
    """

    name: str = Field(min_length=1)
    expose: frozenset[FieldPath] = frozenset()
    frame_window: tuple[NonNegativeInt, NonNegativeInt] | None = None

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.frame_window is not None and self.frame_window[0] >= self.frame_window[1]:
            raise ValueError(f"frame_window {self.frame_window} must satisfy start < end")
        return self

    def apply(self, capture: Capture) -> tuple[Capture, AuditRecord]:
        """The method-visible copy of ``capture`` and a record of what was withheld."""
        full = capture.to_json()
        kept = paths.select(full, self.expose)
        visible = Capture.model_validate(kept)
        if self.frame_window is not None:
            visible = visible.windowed(*self.frame_window)
        audit = AuditRecord(
            policy=self.name,
            capture_id=capture.capture_id,
            hidden=tuple(paths.removed(full, kept)),
            frame_window=self.frame_window,
        )
        return visible, audit


CALIBRATED = VisibilityPolicy(
    name="calibrated",
    expose=frozenset(
        {
            "source",
            "views.*.intrinsics",
            "views.*.extrinsics",
            "views.*.extrinsics_per_frame",
            "world",
            "scene",
            "text",
        }
    ),
)
"""Video plus calibrated cameras, metric scale, scene objects and text."""

VIDEO_ONLY = VisibilityPolicy(name="video_only", expose=frozenset({"source", "text"}))
"""Only the video and the text description."""
