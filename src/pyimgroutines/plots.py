from __future__ import annotations
from typing import Iterable
import pyqtgraph as pg
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import QPointF, Qt, QRectF
from PySide6.QtWidgets import QApplication

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

# Wrapper around default PlotItem with extra references
class PgPlotItem:
    trackingDtype = np.float64

    CURSOR_SHOW_NONE = 0
    CURSOR_SHOW_POS = 1
    CURSOR_SHOW_VALUE = 2

    def __init__(self, plotItem: pg.PlotItem, parent: PgFigure):
        self._plotItem = plotItem
        self._parent = parent
        # Extras
        self._cursorMode = self.CURSOR_SHOW_POS
        self._im = pg.ImageItem()
        self._cursorPos = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._btmLeftPos = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._pixelSize = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._mouseLabel = pg.TextItem()
        self._imgData = np.empty((0, 0), dtype=np.float32)
        self._cbar = pg.ColorBarItem()

    # Forward everything unknown to the original PlotItem
    def __getattr__(self, name):
        return getattr(self._plotItem, name)

    @property
    def base(self) -> pg.PlotItem:
        """
        Return the underlying, original pyqtgraph PlotItem class.
        """
        return self._plotItem

    @property
    def im(self) -> pg.ImageItem:
        return self._im

    @property
    def cbar(self) -> pg.ColorBarItem:
        return self._cbar

    @property
    def imgData(self) -> np.ndarray:
        return self._imgData


    def image(
        self,
        arr: np.ndarray,
        xywh: list | None = None,
        addHalfPixelBorder: bool = True,
        zvalue: int = -100,
        colorbar: bool = True,
        includeLegend: bool = False
    ):
        if includeLegend:
            self.addLegend()

        # Save reference to image data for this subplot
        self._imgData = arr
        self._im = pg.ImageItem(arr, axisOrder='row-major') # default to row-major instead

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
            self._cbar = self.addColorBar(self._im, colorMap=cm2use)

        self.addItem(self._im)

        # Slot to show cursor position
        self._mouseLabel = pg.TextItem(text="", anchor=(0, 1)) # show to top right of cursor
        # self._mouseLabel.setFlag(self._mouseLabel.GraphicsItemFlag.ItemIgnoresTransformations)
        self.addItem(self._mouseLabel, ignoreBounds=True)

        # Set custom slot for mouseMoved
        self._parent.scene().sigMouseMoved.connect(self._parent.mouseMoved) # pyright: ignore

    def _rotateCursorMode(self):
        self._cursorMode = (self._cursorMode + 1) % 3
        self._setMouseLabelText()

    def _toggleTextColour(self):
        colour = self._mouseLabel.color
        # Invert the colour
        self._mouseLabel.setColor(
            pg.mkColor(
                255 - colour.red(),
                255 - colour.green(),
                255 - colour.blue(),
            )
        )

    def _getNearestImagePointIndex(self) -> np.ndarray | None:
        offset = self._cursorPos - self._btmLeftPos
        index = offset / self._pixelSize # this is in x/y
        dataRows, dataCols = self._imgData.shape
        if np.any(index < 0) or index[0] > dataCols or index[1] > dataRows: # pyright: ignore
            return None
        return index.astype(np.int32)

    def _setMouseLabelText(self):
        if self._cursorMode == self.CURSOR_SHOW_POS:
            self._mouseLabel.setText(f"{self._cursorPos[0]:.6g}, {self._cursorPos[1]:.6g}")
        elif self._cursorMode == self.CURSOR_SHOW_VALUE:
            index = self._getNearestImagePointIndex() # pyright: ignore
            if index is None:
                self._mouseLabel.setText(f"OOB") # pyright: ignore
            else:
                self._mouseLabel.setText(f"{self._imgData[int(index[1]), int(index[0])]}") # pyright: ignore
        else:
            self._mouseLabel.setText("")

    def _setCursorPositionInPlot(self, coords):
        self._cursorPos[0] = coords.x()
        self._cursorPos[1] = coords.y()
        self._mouseLabel.setPos(self._cursorPos[0], self._cursorPos[1])


class PgFigure(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plts = np.empty((0, 0), dtype=PgPlotItem)
        self._currPlotIndex = np.array([0, 0], dtype=np.uint8)
        self.setPlotGrid(1, 1) # Default to having a single plot

    def __getitem__(self, idx: tuple | int) -> PgPlotItem:
        """
        Return the indexed plot from the grid of plots.
        """
        return self._plts[idx]

    @property
    def plt(self) -> PgPlotItem:
        if self._plts.size != 1:
            raise ValueError(".plt can only be used when there is only 1 plot!")
        return self._plts[0,0]

    @property
    def plts(self) -> np.ndarray:
        return self._plts

    def setPlotGrid(self, rows: int, cols: int):
        self.clear() # pyright: ignore
        self._plts = np.empty((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                self._plts[i,j] = PgPlotItem(self.addPlot(row=i, col=j), self) # pyright: ignore

    def keyPressEvent(self, ev):
        plt = self[self._currPlotIndex[0], self._currPlotIndex[1]]
        if ev.key() == Qt.Key.Key_V:
            # Toggle cursor mode between position and value
            plt._rotateCursorMode()
        elif ev.key() == Qt.Key.Key_C:
            # Toggle text colour
            plt._toggleTextColour()
        return super().keyPressEvent(ev)

    def mouseMoved(self, evt: QPointF):
        for i in range(self._plts.shape[0]):
            for j in range(self._plts.shape[1]):
                plt = self._plts[i,j]
                if plt.sceneBoundingRect().contains(evt): # pyright: ignore
                    # Cache that this is the currently hovered plot
                    self._currPlotIndex[:] = [i, j]
                    # print(f"in {i},{j}")
                    coords = plt.vb.mapSceneToView(evt) # pyright: ignore
                    # Update internal cursor position and mouse label position
                    plt._setCursorPositionInPlot(coords)
                    # Update text (using internal cursor positions)
                    plt._setMouseLabelText()
                    return



if __name__ == "__main__":
    import sys
    closeAllFigs()

    # f = PgFigure()
    # f.setPlotGrid(2,3)
    # x = np.arange(6).reshape((2,3))
    # for i in range(2):
    #     for j in range(3):
    #         y = x.copy()
    #         y[i, j] = 10
    #         f.plts[i, j].image(y)
    #
    # f.show()
    #

    if len(sys.argv) > 1:
        length = int(sys.argv[1])
        rows = length
        cols = length + 2
    else:
        rows = 3
        cols = 5
        length = 3
    x = np.arange(rows * cols).reshape((rows, cols))
    f = PgFigure()
    f.plt.image(x)
    f.show()

    if length <= 7:
        f2 = PgFigure()
        y = np.arange(length*length)
        y = y % 2
        y = y.reshape((length, length))
        f2.plt.image(y)
        f2.show()

        f3 = PgFigure()
        z = y.copy().astype(np.float32)
        z[0,0] = np.nan
        z[-1,-1] = np.nan
        f3.plt.image(z, includeLegend = True)
        f3[0,0].plot([1,1],[2,2],name="test?") # Just to show legend
        f3.show()

        f4 = PgFigure()
        f4.setPlotGrid(2,2)
        # print(f4.plt)
        f4.show()
        data = np.arange(2*2).reshape((2,2)).astype(np.float32)
        for (i, j), plt in np.ndenumerate(f4.plts):
            datac = data.copy()
            datac[i,j] = np.nan
            plt.image(datac)
            plt._cbar.setLevels((1,2))


    # if os.name == "nt":
    #     forceShow()

