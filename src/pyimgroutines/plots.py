from typing import Iterable
import pyqtgraph as pg
import numpy as np
from PySide6.QtCore import QPointF, Qt, QRectF
import os

def forceShow():
    app = pg.mkQApp()
    app.exec()

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

    def image(
        self,
        arr: np.ndarray,
        xywh: list | None = None,
        addHalfPixelBorder: bool = True,
        zvalue: int = -100,
        colorbar: bool = True
    ):
        self._im = pg.ImageItem(arr)
        if self._plt is None:
            self._plt = self.addPlot() # pyright: ignore

        # Default to 0,0 bottom left, width and height in pixels
        if xywh is None:
            xywh = [0, 0, arr.shape[1], arr.shape[0]]

        # Centres each pixel on the grid coordinates (instead of corners)
        if addHalfPixelBorder:
            pixelWidth = xywh[2] / arr.shape[1]
            pixelHeight = xywh[3] / arr.shape[0]
            xywh[0] -= 0.5 * pixelWidth
            xywh[1] -= 0.5 * pixelHeight

        self._im.setRect(QRectF(*xywh))
        self._im.setZValue(zvalue)

        cm2use = pg.colormap.getFromMatplotlib("viridis")
        self._im.setLookupTable(cm2use.getLookupTable()) # pyright: ignore

        if colorbar:
            self._plt.addColorBar(self._im, colorMap=cm2use)

        self._plt.addItem(self._im)

        # TODO: maybe make a custom signal/slot?
        self.scene().sigMouseMoved.connect(self.mouseMoved) # pyright: ignore

    def mouseMoved(self, evt: QPointF):
        if self._plt.sceneBoundingRect().contains(evt): # pyright: ignore
            coords = self._plt.vb.mapSceneToView(evt) # pyright: ignore
            print(f"Mouse position: {coords.x()}, {coords.y()}")



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        length = int(sys.argv[1])
    else:
        length = 3
    x = np.arange(length*length).reshape((length, length))
    f = PgFigure()
    f.image(x)
    f.show()

    if length <= 7:
        f2 = PgFigure()
        y = np.arange(length*length)
        y = y % 2
        y = y.reshape((length, length))
        f2.image(y)
        f2.show()

    if os.name == "nt":
        forceShow()

