from typing import Iterable
import pyqtgraph as pg
import numpy as np
from PySide6.QtCore import Qt, QRectF

class PgFigure(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._im = None
        self._plt = None

    @property
    def im(self) -> None | pg.ImageItem:
        return self._im

    @property
    def plt(self) -> None | pg.PlotItem:
        return self._plt

    def image(self, arr: np.ndarray, xywh: Iterable | None = None, zvalue: int = -100, colorbar: bool = True):
        self._im = pg.ImageItem(arr)
        if self._plt is None:
            self._plt = self.addPlot() # pyright: ignore

        if xywh is not None:
            self._im.setRect(QRectF(*xywh))
        self._im.setZValue(zvalue)

        cm2use = pg.colormap.getFromMatplotlib("viridis")
        self._im.setLookupTable(cm2use.getLookupTable()) # pyright: ignore

        if colorbar:
            self._plt.addColorBar(self._im, colorMap=cm2use)

        self._plt.addItem(self._im)

if __name__ == "__main__":
    x = np.arange(9).reshape((3,3))
    f = PgFigure()
    f.image(x)
    f.show()
