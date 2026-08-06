import numpy as np


def bounding_box(
    points: np.ndarray,
    padding_factor: float = 0.0,
    min_span: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the axis-aligned bounding box of 2D points.

    Parameters
    ----------
    points : np.ndarray
        An array of shape ``(N, 2)`` containing x and y coordinates.
    padding_factor : float
        Fraction of each axis span to add on both sides of that axis.
        For example, ``padding_factor=0.25`` adds ``0.25 * span`` below
        the minimum and above the maximum coordinate. Defaults to no padding.
    min_span : float
        Minimum span enforced independently on each axis before padding.
        This helps protect against a zero-length span across a particular
        axis when all points have the same coordinate there. Defaults to
        ``0.05``. Set to ``0`` to disable this behavior.

    Returns
    -------
    tuple
        The bounding box in pyqtgraph view-box format:
        ``((xmin, xmax), (ymin, ymax))``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must have shape (N, 2) and contain at least one point")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    if not np.isfinite(padding_factor) or padding_factor < 0:
        raise ValueError("padding_factor must be a finite, non-negative number")
    if not np.isfinite(min_span) or min_span < 0:
        raise ValueError("min_span must be a finite, non-negative number")

    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, min_span)
    padding = padding_factor * span

    return (
        (lower[0] - padding[0], upper[0] + padding[0]),
        (lower[1] - padding[1], upper[1] + padding[1]),
    )


__all__ = ["bounding_box"]
