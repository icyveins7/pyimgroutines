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
        self._im = None
        # NOTE: where the cursor actually is, not necessarily where the label is
        self._cursorPos = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._btmLeftPos = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._pixelSize = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._mouseLabel = pg.TextItem()
        self._imgData = np.empty((0, 0), dtype=np.float32)
        self._cbar = pg.ColorBarItem()
        self._lockedPointing = False
        self._addHalfPixelBorder = False

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
        if self._im is None:
            raise TypeError("No image item created yet")
        return self._im

    @property
    def cbar(self) -> pg.ColorBarItem:
        return self._cbar

    @property
    def imgData(self) -> np.ndarray:
        return self._imgData

    def rectangle(
        self,
        xy: tuple | list | np.ndarray,
        wh: tuple | list | np.ndarray,
        pen: str = "r"
    ):
        """
        Adds a rectangle via pg.ROI.

        Parameters
        ----------
        xy : tuple | list | np.ndarray
            Bottom-left point.

        wh : tuple | list | np.ndarray
            Width/height.

        pen : str
            Colour of the rectangle border.


        Returns
        -------
        rect : pg.ROI
            Rectangle item.
        """
        # RectROI creates a non-removable scaler at the top right,
        # so we don't use it
        rect = pg.ROI(xy, wh, pen=pg.mkPen(pen), # pyright: ignore
                          movable=False,
                          rotatable=False,
                          resizable=False,
                          antialias=False)
        self.addItem(rect)
        return rect

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
            self._addHalfPixelBorder = addHalfPixelBorder

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

        # Set custom slot for mouseMoved/mouseClicked
        self._parent.scene().sigMouseMoved.connect(self._parent.mouseMoved) # pyright: ignore
        self._parent.scene().sigMouseClicked.connect(self._parent.mouseClicked) # pyright: ignore

    def _rotateCursorMode(self):
        self._cursorMode = (self._cursorMode + 1) % 3
        self._setMouseLabelTextAndPos()

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

    def _getLockedPosition(self):
        index = self._getNearestImagePointIndex()
        if index is None:
            return None, None
        bottomLeftPointerPos = self._btmLeftPos + self._pixelSize * 0.5 * self._addHalfPixelBorder
        return index * self._pixelSize + bottomLeftPointerPos, index

    def _setMouseLabelTextAndPos(self):
        # Set the position of the label
        if self._lockedPointing:
            pos, index = self._getLockedPosition()
        else:
            pos = self._cursorPos
            index = None

        # Only set text if position is valid
        if pos is not None:
            self._mouseLabel.setPos(pos[0], pos[1])

            # Set the text of the label
            if self._cursorMode == self.CURSOR_SHOW_POS:
                self._mouseLabel.setText(f"{pos[0]:.6g}, {pos[1]:.6g}")
            elif self._cursorMode == self.CURSOR_SHOW_VALUE:
                index = self._getNearestImagePointIndex() # pyright: ignore
                if index is None:
                    self._mouseLabel.setText(f"OOB") # pyright: ignore
                else:
                    self._mouseLabel.setText(f"[{int(index[1])}, {int(index[0])}]: {self._imgData[int(index[1]), int(index[0])]}") # pyright: ignore
            else:
                self._mouseLabel.setText("")
        else:
            self._mouseLabel.setText("")


    def _setCursorPositionInPlot(self, coords):
        self._cursorPos[0] = coords.x()
        self._cursorPos[1] = coords.y()

    def _toggleImage(self):
        if self._im is not None:
            self._im.setVisible(
                not self._im.isVisible()
            )

    def _toggleLockedPointing(self):
        self._lockedPointing = not self._lockedPointing
        # Re-render the text labels
        self._setMouseLabelTextAndPos()


class PgFigure(pg.GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plts = np.empty((0, 0), dtype=PgPlotItem)
        self._currPlotIndex = np.array([0, 0], dtype=np.uint8)
        self.setPlotGrid(1, 1) # Default to having a single plot

        self._isMaximized = False

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

    def setPlotGrid(
        self,
        rows: int,
        cols: int,
        linkX: bool = False,
        linkY: bool = False,
        aspectLocked: bool = False
    ):
        self.clear() # pyright: ignore
        self._plts = np.empty((rows, cols), dtype=object)
        for i, j in np.ndindex(self._plts.shape):
            plt = PgPlotItem(self.addPlot(row=i, col=j), self) # pyright: ignore
            if aspectLocked:
                plt.setAspectLocked()
            if linkX and (i + j > 0):
                plt.setXLink(self._plts[0, 0])
            if linkY and (i + j > 0):
                plt.setYLink(self._plts[0, 0])
            # TODO: if both links specified then enable across subplot mouse tracking
            self._plts[i, j] = plt

    def keyPressEvent(self, ev):
        plt = self[self._currPlotIndex[0], self._currPlotIndex[1]]
        if ev.key() == Qt.Key.Key_V:
            # Toggle cursor mode between position and value
            plt._rotateCursorMode()
        elif ev.key() == Qt.Key.Key_C:
            # Toggle text colour
            plt._toggleTextColour()
        elif ev.key() == Qt.Key.Key_Escape and self._isMaximized:
            self.subplotMinimize()
        elif ev.key() == Qt.Key.Key_I:
            plt._toggleImage()
        elif ev.key() == Qt.Key.Key_L:
            plt._toggleLockedPointing()
        return super().keyPressEvent(ev)

    def subplotMaximize(self):
        i, j = self._currPlotIndex
        plt = self[i,j].base
        # wipe layout, items are still cached
        self.clear() # pyright:ignore
        self.addItem(plt, row=0, col=0) # pyright: ignore
        # TODO: for any plot other than 0,0 it seems the xlim/ylim is not retained
        self._isMaximized = True

    def subplotMinimize(self):
        self.clear() # pyright:ignore
        for (i, j), plt in np.ndenumerate(self._plts):
            self.addItem(plt.base, row=i, col=j) # pyright:ignore
        self._isMaximized = False

    def mouseClicked(self, evt):
        if evt.double() and not self._isMaximized and self._plts.size > 1:
            self.subplotMaximize()

    def mouseMoved(self, evt: QPointF):
        for i in range(self._plts.shape[0]):
            for j in range(self._plts.shape[1]):
                plt = self._plts[i,j]
                if plt.sceneBoundingRect().contains(evt): # pyright: ignore
                    # Cache that this is the currently hovered plot
                    self._currPlotIndex[:] = [i, j]
                    # print(f"in {i},{j}")
                    coords = plt.vb.mapSceneToView(evt) # pyright: ignore
                    # Cache internal cursor position
                    plt._setCursorPositionInPlot(coords)
                    # Update text and position
                    plt._setMouseLabelTextAndPos()
                    return



if __name__ == "__main__":
    import sys
    closeAllFigs()

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
    f.plt.rectangle((-1,-1), (cols + 1, rows + 1))
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
        f4.setPlotGrid(2,2,True,True,True)
        # print(f4.plt)
        f4.show()
        data = np.arange(2*2).reshape((2,2)).astype(np.float32)
        for (i, j), plt in np.ndenumerate(f4.plts):
            datac = data.copy()
            datac[i,j] = np.nan
            plt.image(datac)
            plt.cbar.setLevels((1,2))


    # if os.name == "nt":
    #     forceShow()

