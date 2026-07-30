from collections.abc import Iterable

import numpy as np


def points_to_image(
    points: np.ndarray,
    img_dims: Iterable[int] = (2048, 2048),
    img_xywh: Iterable[float] | None = None,
    dtype: type = np.uint8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize point coordinates into a 2D occupancy/count image.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (N, 2), with columns containing x and y coordinates.

    img_dims : tuple[int, int]
        Image dimensions as (height, width). Defaults to (2048, 2048).

    img_xywh : np.ndarray | None
        Image bounds as (x0, y0, width, height), where (x0, y0) is the
        bottom-left point. If None, the bounds are taken from the point data.

    dtype : type
        Integer dtype of the returned image. Counts are accumulated internally
        as uint32, then scaled so the maximum count maps to the maximum value
        of this dtype. Defaults to np.uint8.

    Returns
    -------
    image : np.ndarray
        An image of shape (height, width) with the requested dtype. The
        accumulated counts are linearly scaled so the maximum count maps to
        the maximum value of the dtype. Row zero corresponds to the lower y
        bound of img_xywh.

    img_xywh : np.ndarray
        The image bounds used for rasterization as (x0, y0, width, height).
        Automatically generated bounds include the half-pixel shift required
        for plotting pixel centres, so pass ``addHalfPixelBorder=False`` when
        passing the returned bounds to ``PgPlotItem.image()``.
    """
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError("dtype must be an integer dtype")

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(points) == 0:
        raise ValueError("points must contain at least one point")

    img_dims = np.asarray(tuple(img_dims))
    if img_dims.shape != (2,) or not np.issubdtype(img_dims.dtype, np.integer):
        raise ValueError("img_dims must contain two integer dimensions")
    if np.any(img_dims <= 0):
        raise ValueError("img_dims must contain positive dimensions")
    img_height, img_width = img_dims.astype(np.intp)

    if img_xywh is None:
        if img_width < 2 or img_height < 2:
            raise ValueError("auto-generated bounds require image dimensions >= 2")
        xy_min = np.min(points, axis=0)
        xy_max = np.max(points, axis=0)
        xy_span = xy_max - xy_min

        # Treat the minimum and maximum coordinates as pixel centres. The
        # spacing is therefore based on (dimension - 1), not dimension, and
        # the image extent includes one pixel spacing beyond the data span;
        # half a pixel on both ends.
        auto_pixel_width = xy_span[0] / (img_width - 1)
        auto_pixel_height = xy_span[1] / (img_height - 1)
        img_xywh = np.array([
            xy_min[0],
            xy_min[1],
            xy_span[0] + auto_pixel_width,
            xy_span[1] + auto_pixel_height,
        ])
    else:
        img_xywh = np.asarray(tuple(img_xywh), dtype=float)
        if img_xywh.shape != (4,):
            raise ValueError("img_xywh must have shape (4,)")

    x0, y0, width, height = img_xywh
    if width <= 0 or height <= 0:
        raise ValueError("img_xywh width and height must be positive")

    pixel_width = width / img_width
    pixel_height = height / img_height

    # Treat the input coordinates as pixel centres, matching PgPlotItem.image:
    # shift the image's lower-left corner down and left by half a pixel.
    img_xywh[0] -= 0.5 * pixel_width
    img_xywh[1] -= 0.5 * pixel_height
    shifted_x0, shifted_y0 = img_xywh[:2]

    inside = (
        (points[:, 0] >= x0)
        & (points[:, 0] <= x0 + width)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y0 + height)
    )
    insidepoints = points[inside]

    # Each pixel represents a half-open interval in image coordinates:
    # [shifted_x0 + i * pixel_width, shifted_x0 + (i + 1) * pixel_width)
    # and similarly for y. Floor maps a point to the pixel containing it.
    # The upper/right boundary is inclusive in the input bounds, so points
    # exactly at x0 + width or y0 + height are clipped into the final pixel.
    x_indices = np.floor((insidepoints[:, 0] - shifted_x0) / pixel_width).astype(np.intp)
    y_indices = np.floor((insidepoints[:, 1] - shifted_y0) / pixel_height).astype(np.intp)
    x_indices = np.clip(x_indices, 0, img_width - 1)
    y_indices = np.clip(y_indices, 0, img_height - 1)

    image = np.zeros((img_height, img_width), dtype=np.uint32)
    # np.add.at is required rather than image[y_indices, x_indices] += 1:
    # multiple insidepoints may map to one pixel, and add.at accumulates every
    # repeated index instead of buffering the advanced-indexing update.
    np.add.at(image, (y_indices, x_indices), 1)

    max_count = image.max()
    if max_count > 0:
        image = (image / max_count * np.iinfo(dtype).max).astype(dtype)
    else:
        image = image.astype(dtype)

    return image, img_xywh
