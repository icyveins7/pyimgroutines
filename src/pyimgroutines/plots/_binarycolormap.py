import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor


def makeBinaryColormap(
    offColor: QColor,
    onColor: QColor,
    ensureTransparentZero: bool = True,
) -> pg.ColorMap:
    """
    Create a color map between off and on colors.

    Parameters
    ----------
    offColor : QColor
        Color at the lower end of the gradient.

    onColor : QColor
        Color at the upper end of the gradient.

    ensureTransparentZero : bool, default True
        Make the value at exactly zero transparent while keeping every value
        above zero opaque. If ``False``, interpolate directly between
        `offColor` and `onColor`, preserving the original behavior.
        You usually want this so that layering multiple images doesn't end up blocking
        each other when a pixel has zero count.

    Returns
    -------
    pg.ColorMap
        Color map spanning the normalized range from zero to one.
    """
    if not ensureTransparentZero:
        return pg.ColorMap(pos=[0, 1], color=[offColor, onColor])

    # For completely 0 count pixels we make it fully transparent
    transparentOffColor = QColor(offColor)
    transparentOffColor.setAlpha(0)
    # This is where the gradient actually starts
    opaqueOffColor = QColor(offColor)
    opaqueOffColor.setAlpha(255)
    opaqueOnColor = QColor(onColor)
    opaqueOnColor.setAlpha(255)
    # Use the first float64 value after 0 as the 'limit' at which the opaque starts
    firstPositive = np.nextafter(0.0, 1.0)
    return pg.ColorMap(
        pos=[0.0, firstPositive, 1.0],
        color=[transparentOffColor, opaqueOffColor, opaqueOnColor],
    )
