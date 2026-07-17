from PySide6.QtGui import QColor
import pyqtgraph as pg

def makeBinaryColormap(offColor: QColor, onColor: QColor) -> pg.ColorMap:
    return pg.ColorMap(pos=[0, 1], color=[offColor, onColor])
