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

        self._cursorMode = self.CURSOR_SHOW_POS
        self._trackingDtype = trackingDtype

        # Plot-specific
        self._im = None
        self._plt = None
        self._cursorPos = np.array([0, 0], dtype=trackingDtype)
        self._cursorPosPlotIndex = np.array([0, 0], dtype=np.int32)
        self._btmLeftPos = np.array([np.nan, np.nan], dtype=trackingDtype)
        self._pixelSize = np.array([np.nan, np.nan], dtype=trackingDtype)
        self._mouseLabel = None
        self._imgData = None
        self._cbar = None

    @property
    def im(self) -> None | pg.ImageItem:
        return self._im

    def __getitem__(self, idx: tuple | int):
        """
        Return the plot from the grid of plots.
        """
        if self._plt is None:
            raise TypeError("No plots currently set")

        if isinstance(idx, int):
            if len(self._plt) > 1:
                raise IndexError("More than 1 row of plots, invalid single index")
            return self._plt[0][idx]
        elif isinstance(idx, tuple):
            return self._plt[idx[0]][idx[1]]

    @property
    def plt(self) -> list[list[pg.PlotItem]]:
        if self._plt is None:
            raise TypeError("Plots have not yet been added")
        return self._plt

    def setPlotGrid(self, rows: int, cols: int):
        # Reset all plot-related things
        self._plt = list()
        self._imgData = list()
        self._im = list()
        self._pixelSize = list()
        self._btmLeftPos = list()
        self._cursorPos = list()
        self._mouseLabel = list()
        self._cbar = list()

        for i in range(rows):
            self._plt.append(list())
            self._imgData.append(list())
            self._im.append(list())
            self._pixelSize.append(list())
            self._btmLeftPos.append(list())
            self._cursorPos.append(list())
            self._mouseLabel.append(list())
            self._cbar.append(list())

            for j in range(cols):
                self._plt[-1].append(
                    self.addPlot(row=i, col=j) # pyright:ignore
                )
                self._imgData[-1].append(None)
                self._im[-1].append(None)
                self._pixelSize[-1].append(np.array([np.nan, np.nan], dtype=self._trackingDtype))
                self._btmLeftPos[-1].append(np.array([np.nan, np.nan], dtype=self._trackingDtype))
                self._cursorPos[-1].append(np.array([0, 0], dtype=self._trackingDtype))
                self._mouseLabel[-1].append(None)
                self._cbar[-1].append(None)

    def image(
        self,
        arr: np.ndarray,
        xywh: list | None = None,
        addHalfPixelBorder: bool = True,
        zvalue: int = -100,
        colorbar: bool = True,
        pltIdx: tuple[int, int] | None = None,
        includeLegend: bool = False
    ):
        if pltIdx is None:
            self.setPlotGrid(1, 1)
            plti = 0
            pltj = 0
        else:
            plti, pltj = pltIdx

        plt = self.plt[plti][pltj]

        if includeLegend:
            plt.addLegend()

        # Save reference to image data for this subplot
        self._imgData[plti][pltj] = arr
        self._im[plti][pltj] = pg.ImageItem(arr, axisOrder='row-major') # default to row-major instead

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
        self._btmLeftPos[plti][pltj][0] = xywh[0]
        self._btmLeftPos[plti][pltj][1] = xywh[1]
        self._pixelSize[plti][pltj][0] = pixelWidth
        self._pixelSize[plti][pltj][1] = pixelHeight

        self._im[plti][pltj].setRect(QRectF(*xywh))
        self._im[plti][pltj].setZValue(zvalue)

        cm2use = pg.colormap.getFromMatplotlib("viridis")
        self._im[plti][pltj].setLookupTable(cm2use.getLookupTable()) # pyright: ignore

        if colorbar:
            self._cbar[plti][pltj] = plt.addColorBar(self._im[plti][pltj], colorMap=cm2use)

        plt.addItem(self._im[plti][pltj])

        # Set custom slot for mouseMoved
        self.scene().sigMouseMoved.connect(self.mouseMoved) # pyright: ignore

        # Slot to show cursor position
        self._mouseLabel[plti][pltj] = pg.TextItem(text="", anchor=(0, 1)) # show to top right of cursor
        # self._mouseLabel.setFlag(self._mouseLabel.GraphicsItemFlag.ItemIgnoresTransformations)
        plt.addItem(self._mouseLabel[plti][pltj], ignoreBounds=True)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_V:
            # Toggle cursor mode between position and value
            self._cursorMode = (self._cursorMode + 1) % 3
            # Update now
            self._setMouseLabelText()
        elif ev.key() == Qt.Key.Key_C:
            # Toggle text colour
            self._toggleTextColour()
        return super().keyPressEvent(ev)

    def _toggleTextColour(self):
        plti, pltj = self._cursorPosPlotIndex
        colour = self._mouseLabel[plti][pltj].color
        # Invert the colour
        self._mouseLabel[plti][pltj].setColor(
            pg.mkColor(
                255 - colour.red(),
                255 - colour.green(),
                255 - colour.blue(),
            )
        )

    def _getNearestImagePointIndex(self) -> np.ndarray | None:
        plti, pltj = self._cursorPosPlotIndex
        offset = self._cursorPos[plti][pltj] - self._btmLeftPos[plti][pltj]
        index = offset / self._pixelSize[plti][pltj] # this is in x/y
        dataRows, dataCols = self._imgData[plti][pltj].shape
        if np.any(index < 0) or index[0] > dataCols or index[1] > dataRows: # pyright: ignore
            return None
        return index.astype(np.int32)

    def _setMouseLabelText(self):
        plti, pltj = self._cursorPosPlotIndex
        if self._cursorMode == self.CURSOR_SHOW_POS:
            self._mouseLabel[plti][pltj].setText(f"{self._cursorPos[plti][pltj][0]:.4g}, {self._cursorPos[plti][pltj][1]:.4g}")
        elif self._cursorMode == self.CURSOR_SHOW_VALUE:
            index = self._getNearestImagePointIndex() # pyright: ignore
            if index is None:
                self._mouseLabel[plti][pltj].setText(f"OOB") # pyright: ignore
            else:
                self._mouseLabel[plti][pltj].setText(f"{self._imgData[plti][pltj][int(index[1]), int(index[0])]}") # pyright: ignore
        else:
            self._mouseLabel[plti][pltj].setText("")

    def _setCursorPositionInPlot(self, coords, plti, pltj):
        self._cursorPosPlotIndex[0] = plti
        self._cursorPosPlotIndex[1] = pltj
        self._cursorPos[plti][pltj][0] = coords.x()
        self._cursorPos[plti][pltj][1] = coords.y()

    def mouseMoved(self, evt: QPointF):
        for i, rowPlt in enumerate(self.plt):
            for j, plt in enumerate(rowPlt):
                if plt.sceneBoundingRect().contains(evt): # pyright: ignore
                    # print(f"in {i},{j}")
                    coords = plt.vb.mapSceneToView(evt) # pyright: ignore
                    # Update internal cursor position
                    self._setCursorPositionInPlot(coords, i, j)
                    # print(f"Mouse position: {coords.x()}, {coords.y()}")
                    # Tracking mouse label
                    self._mouseLabel[i][j].setPos(coords.x(), coords.y())
                    # Update text (using internal cursor positions)
                    self._setMouseLabelText()

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
        f3.image(z, includeLegend = True)
        f3._plt[0][0].plot([1,1],[2,2],name="test?") # Just to show legend
        f3.show()

        f4 = PgFigure()
        f4.setPlotGrid(2,2)
        # print(f4.plt)
        f4.show()
        data = np.arange(2*2).reshape((2,2)).astype(np.float32)
        for i, row in enumerate(f4.plt):
            for j, plt in enumerate(row):
                datac = data.copy()
                datac[i,j] = np.nan
                f4.image(datac, pltIdx = (i, j))
                f4._cbar[i][j].setLevels((1,2))




    # if os.name == "nt":
    #     forceShow()

