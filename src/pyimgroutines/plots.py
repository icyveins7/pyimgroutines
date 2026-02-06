from typing import Iterable
import pyqtgraph as pg
import numpy as np
from PySide6.QtCore import QPointF, Qt, QRectF
from PySide6.QtWidgets import QApplication
import os

def closeAllFigs():
    QApplication.closeAllWindows()

def forceShow():
    """
    Forces the GUI to show, which can help when .show() doesn't work.
    Usually, .show() should work, even in ipython. Remember to call
    %gui qt in ipython so that the GUI will not block the interpreter.
    """
    app = pg.mkQApp()
    app.exec()

class PgFigure(pg.GraphicsLayoutWidget):
    CURSOR_SHOW_NONE = 0
    CURSOR_SHOW_POS = 1
    CURSOR_SHOW_VALUE = 2

    def __init__(self, *args, trackingDtype: type = np.float64, **kwargs):
        super().__init__(*args, **kwargs)

        self._im = None
        self._plt = None
        self._cursorMode = self.CURSOR_SHOW_POS
        self._cursorPos = np.array([0, 0], dtype=trackingDtype)
        self._btmLeftPos = np.array([np.nan, np.nan], dtype=trackingDtype)
        self._pixelSize = np.array([np.nan, np.nan], dtype=trackingDtype)
        self._imgData = None
        self._cbar = None

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
        self._imgData = arr
        self._im = pg.ImageItem(arr)
        if self._plt is None:
            self._plt = self.addPlot() # pyright: ignore

        # Default to 0,0 bottom left, width and height in pixels
        if xywh is None:
            xywh = [0, 0, arr.shape[1], arr.shape[0]]

        # Centres each pixel on the grid coordinates (instead of corners)
        pixelWidth = 1
        pixelHeight = 1
        if addHalfPixelBorder:
            pixelWidth = xywh[2] / arr.shape[1]
            pixelHeight = xywh[3] / arr.shape[0]
            xywh[0] -= 0.5 * pixelWidth
            xywh[1] -= 0.5 * pixelHeight

        # Cache these for mouse-over mechanics
        self._btmLeftPos[0] = xywh[0]
        self._btmLeftPos[1] = xywh[1]
        self._pixelSize[0] = pixelWidth
        self._pixelSize[1] = pixelHeight

        self._im.setRect(QRectF(*xywh))
        self._im.setZValue(zvalue)

        cm2use = pg.colormap.getFromMatplotlib("viridis")
        self._im.setLookupTable(cm2use.getLookupTable()) # pyright: ignore

        if colorbar:
            self._cbar = self._plt.addColorBar(self._im, colorMap=cm2use)

        self._plt.addItem(self._im)

        # Set custom slot for mouseMoved
        self.scene().sigMouseMoved.connect(self.mouseMoved) # pyright: ignore

        # Slot to show cursor position
        self._mouseLabel = pg.TextItem(text="", anchor=(0, 1)) # show to top right of cursor
        # self._mouseLabel.setFlag(self._mouseLabel.GraphicsItemFlag.ItemIgnoresTransformations)
        self._plt.addItem(self._mouseLabel, ignoreBounds=True)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_V:
            # Toggle cursor mode between position and value
            self._cursorMode = (self._cursorMode + 1) % 3
            # Update now
            self._setMouseLabelText()
        return super().keyPressEvent(ev)

    def _getNearestImagePointIndex(self) -> np.ndarray | None:
        offset = self._cursorPos - self._btmLeftPos
        index = offset / self._pixelSize
        if np.any(index < 0) or np.any(index >= self._imgData.shape): # pyright: ignore
            return None
        return index.astype(np.int32)

    def _setMouseLabelText(self):
        if self._cursorMode == self.CURSOR_SHOW_POS:
            self._mouseLabel.setText(f"{self._cursorPos[0]:.4g}, {self._cursorPos[1]:.4g}")
        elif self._cursorMode == self.CURSOR_SHOW_VALUE:
            index = self._getNearestImagePointIndex() # pyright: ignore
            if index is None:
                self._mouseLabel.setText(f"OOB") # pyright: ignore
            else:
                self._mouseLabel.setText(f"{self._imgData[int(index[1]), int(index[0])]}") # pyright: ignore
        else:
            self._mouseLabel.setText("")

    def mouseMoved(self, evt: QPointF):
        if self._plt.sceneBoundingRect().contains(evt): # pyright: ignore
            coords = self._plt.vb.mapSceneToView(evt) # pyright: ignore
            # Update internal cursor position
            self._cursorPos[0] = coords.x()
            self._cursorPos[1] = coords.y()
            # print(f"Mouse position: {coords.x()}, {coords.y()}")
            # Tracking mouse label
            self._mouseLabel.setPos(coords.x(), coords.y())
            # Update text (using internal cursor positions)
            self._setMouseLabelText()



if __name__ == "__main__":
    import sys
    closeAllFigs()
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

        f3 = PgFigure()
        z = y.copy().astype(np.float32)
        z[0,0] = np.nan
        z[-1,-1] = np.nan
        f3.image(z)
        f3.show()

    # if os.name == "nt":
    #     forceShow()

