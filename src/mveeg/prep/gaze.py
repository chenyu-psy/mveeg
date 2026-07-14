"""Canonical gaze geometry used by degree-based artifact rules."""

from __future__ import annotations

from math import isfinite, radians, tan
from numbers import Real


def normalize_gaze_geometry(
    *,
    viewing_distance_cm: float,
    screen_width_cm: float,
    screen_width_px: int,
) -> dict[str, float | int]:
    """Validate and normalize the geometry needed for visual-angle conversion."""

    distance = _positive_number(viewing_distance_cm, "viewing_distance_cm")
    width = _positive_number(screen_width_cm, "screen_width_cm")
    pixels = _positive_number(screen_width_px, "screen_width_px")
    if pixels != round(pixels):
        raise ValueError("screen_width_px must be a positive integer.")
    return {
        "viewing_distance_cm": distance,
        "screen_width_cm": width,
        "screen_width_px": int(pixels),
    }


def _degrees_to_pixels(
    degrees: float,
    gaze_geometry: dict[str, float | int],
) -> int:
    """Convert positive visual degrees to pixels using canonical geometry."""

    degrees = _positive_number(degrees, "visual-angle threshold")
    geometry = normalize_gaze_geometry(**gaze_geometry)
    visual_width_cm = 2 * geometry["viewing_distance_cm"] * tan(
        0.5 * radians(degrees)
    )
    return round(
        visual_width_cm
        / (geometry["screen_width_cm"] / geometry["screen_width_px"])
    )


def _positive_number(value: object, name: str) -> float:
    """Return one finite positive real value without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive number.")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return numeric
