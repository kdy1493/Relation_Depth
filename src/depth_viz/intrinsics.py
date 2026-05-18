"""Camera intrinsics for depth → point cloud back-projection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole camera: fx, fy, cx, cy for image size (width, height)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, width: int, height: int) -> PinholeIntrinsics:
        if width == self.width and height == self.height:
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        return PinholeIntrinsics(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )


# SynTable / NVIDIA Isaac Sim defaults (640×480 render resolution).
_SYNTABLE_FOCAL_LENGTH = 1.88
_SYNTABLE_HORIZ_APERTURE = 2.6327803436685087
_SYNTABLE_VERT_APERTURE = 1.9573321100745658
_SYNTABLE_BASE_W = 640
_SYNTABLE_BASE_H = 480
_SYNTABLE_FX = _SYNTABLE_FOCAL_LENGTH * _SYNTABLE_BASE_W / _SYNTABLE_HORIZ_APERTURE
_SYNTABLE_FY = _SYNTABLE_FOCAL_LENGTH * _SYNTABLE_BASE_H / _SYNTABLE_VERT_APERTURE


def syntable_intrinsics(width: int = _SYNTABLE_BASE_W, height: int = _SYNTABLE_BASE_H) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        width=width,
        height=height,
        fx=_SYNTABLE_FX,
        fy=_SYNTABLE_FY,
        cx=_SYNTABLE_BASE_W / 2.0,
        cy=_SYNTABLE_BASE_H / 2.0,
    )
