"""Image rendering that masks invalid values without changing the source array."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QImage


class _ValidLevelsImageItem(pg.ImageItem):
    """
    Render pixels outside an inclusive valid range as transparent.

    Use this for integer images with reserved invalid values, such as
    ``int32`` images where the maximum integer is a sentinel. The original
    image retains its dtype and values; only the alpha channel of the rendered
    QImage is changed. The valid range is independent of the display color
    levels, so invalid pixels remain transparent when those levels change.

    Parameters
    ----------
    image : np.ndarray
        Source image, retained without modifying its data.
    validLevels : tuple[int, int]
        Inclusive range of values to render normally.
    **kwargs
        Additional arguments passed to ``pg.ImageItem``.
    """

    def __init__(self, image: np.ndarray, validLevels: tuple[int, int], **kwargs):
        self._validLevels = validLevels
        super().__init__(image, **kwargs)

    def render(self):
        super().render()
        if self._unrenderable:
            return

        lower, upper = self._validLevels
        invalid = ~((self.image >= lower) & (self.image <= upper))
        if not np.any(invalid):
            return

        qimage = self.qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.ndarray(
            (qimage.height(), qimage.bytesPerLine()), dtype=np.uint8, buffer=qimage.bits()
        )
        pixels[:, :qimage.width() * 4].reshape(*invalid.shape, 4)[..., 3][invalid] = 0
        self.qimage = qimage
