from __future__ import annotations
from typing import Iterable
from PySide6.QtGui import QPen
import pyqtgraph as pg
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import QPointF, Qt, QRectF
from PySide6.QtWidgets import QApplication
from itertools import repeat

from .customitems import EllipseItem

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
    def mouseLabel(self) -> pg.TextItem:
        return self._mouseLabel

    @property
    def imgData(self) -> np.ndarray:
        return self._imgData

    def ellipses(
        self,
        xy_rxry: np.ndarray,
        pen:  QPen | Iterable[QPen] = pg.mkPen("r")
    ) -> EllipseItem:
        """
        Creates an EllipseItem and adds one or more ellipses to it,
        then adds it to the plot.

        If you need to add more ellipses later, you should directly call
        the .addEllipse() or .addCircle() methods on the returned item.

        Parameters
        ----------
        xy_rxry : np.ndarray
            1D (single ellipse) or 2D (one ellipse per row) array.
            Each ellipse is specified by the centre position (x,y)
            followed by the radii (rx, ry).

        pen : QPen | Iterable[QPen]
            The colour of the border(s) of the ellipse(s).
            If a single QPen is specified, it is used for all the ellipses.
            Otherwise, each QPen is used for its matching ellipse.

        Returns
        -------
        item : EllipseItem
            Returned EllipseItem. Can be used directly to add more ellipses,
            since it is already added to the plot.
        """
        if xy_rxry.ndim == 1:
            xy_rxry = xy_rxry.reshape((1, 4)) # single-row 2d array

        if isinstance(pen, QPen):
            pen = repeat(pen)

        # Create new item if not supplied
        item = EllipseItem()
        for i, (ellipse, ellipsePen) in enumerate(zip(xy_rxry, pen)):
            item.addEllipse(
                ellipse, ellipsePen,
                i == len(xy_rxry) - 1 # update on last one
            )
        self.addItem(item)
        return item

    def circles(self, xy_r: np.ndarray, pen: QPen | Iterable[QPen]) -> EllipseItem:
        """
        Creates an EllipseItem and adds one or more circles to it,
        then adds it to the plot.

        If you need to add more ellipses later, you should directly call
        the .addEllipse() or .addCircle() methods on the returned item.

        Parameters
        ----------
        xy_r : np.ndarray
            1D (single circle) or 2D (one circle per row) array.
            Each circle is specified by the centre position (x,y)
            followed by the radius (r).

        pen : QPen | Iterable[QPen]
            The colour of the border(s) of the circle(s).
            If a single QPen is specified, it is used for all the circles.

        Returns
        -------
        item : EllipseItem
            Returned EllipseItem. Can be used directly to add more circles,
            since it is already added to the plot.
        """
        if xy_r.ndim == 1:
            xy_r = xy_r.reshape((1, 3)) # single-row 2d array

        if isinstance(pen, QPen):
            pen = repeat(pen)

        # Create new item if not supplied
        item = EllipseItem()
        for i, (circle, circlePen) in enumerate(zip(xy_r, pen)):
            item.addCircle(
                circle, circlePen,
                i == len(xy_r) - 1 # update on last one
            )
        self.addItem(item)
        return item


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
        # TODO: change to create plain rectangles? maybe more lightweight than ROI

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
        """
        Plots an image from a numpy array.
        Creates a pg.ImageItem and adds it to the plot, with sensible defaults.
        The expected convention is y-axis upwards, as is created by PgFigure.

        Parameters
        ----------
        arr : np.ndarray
            Input data for the image.

        xywh : list | None
            Bottom-left position (x/y) and width/height (w/h) in a list.
            Defaults to None, which simply uses (0, 0) and the number of pixels respectively.

        addHalfPixelBorder : bool
            Whether to automatically add half a pixel as a border around the entire image.
            This ensures that pixels are centred on the points, rather than having the pixels
            be vertices.

            Example: a 2x2 matrix will extend from -0.5 to +1.5 pixels instead of 0 to 2.

            Defaults to True.

        zvalue : int
            Priority/height of the image. Lower numbers will paint the image 'behind' other items.

        colorbar : bool
            Whether to include a colorbar. Defaults to True.

        includeLegend : bool
            Whether to include a legend. The image item itself is not added to the legend,
            but this is useful so other traditional plot items can be added and show up.
        """
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
        # NOTE: cursorPos may be nan/invalid if hovering over another subplot
        if np.all(np.isnan(self._cursorPos)):
            return None
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
    """
    A pyqtgraph 'figure', built on a subclass around a GraphicsLayoutWidget,
    and internal PgPlotItems, which are subclasses around pg.PlotItem.

    General use-case is to create this, and then use .setPlotGrid() if multiple subplots are required,
    similar to matplotlib's plt.subplots().

    Each subplot can be accessed via the index operator (or via .plt if there is only 1 subplot).

    Examples:
        fig = PgFigure()
        fig.plt.image(...)

        mfig = PgFigure()
        mfig.setPlotGrid(2, 2) # 2x2 subplots
        mfig[0,0].image(...) # Plots at the top-left
        mfig[1,1].image(...) # Plots at the bottom-right

    Additional features:
        - In-built cursor tracking (Hotkey V to toggle modes)
        - In-built subplot minimization/maximization (Double-click/Esc)
        - In-built pixel magnetization (Hotkey L)
    """
    FIGURE_INDEX = 1

    def __init__(self, *args, **kwargs):
        """
        Instantiate a new PgFigure, with a default single subplot.

        All input arguments are passed to pg.GraphicsLayoutWidget().
        """
        self._figIndex = self._getFigureIndex()
        if "title" not in kwargs:
            kwargs["title"] = f"PgFigure_{self._figIndex}"
        super().__init__(*args, **kwargs)
        self._plts = np.empty((0, 0), dtype=PgPlotItem)
        self._currPlotIndex = np.array([0, 0], dtype=np.uint8)
        self.setPlotGrid(1, 1) # Default to having a single plot

        self._isMaximized = False

    def _getFigureIndex(self) -> int:
        index = PgFigure.FIGURE_INDEX
        # Increment static var for next figure
        PgFigure.FIGURE_INDEX += 1
        return index

    def __getitem__(self, idx: tuple | int) -> PgPlotItem:
        """
        Return the indexed plot from the grid of plots.
        """
        return self._plts[idx]

    @property
    def plt(self) -> PgPlotItem:
        """
        Returns the lone PgPlotItem (subplot).
        This is primarily used as syntactic sugar when there is only 1 subplot,
        although indexing (e.g. fig[0,0]) still works.

        Throws an error if there are multiple subplots.
        """
        if self._plts.size != 1:
            raise ValueError(".plt can only be used when there is only 1 plot!")
        return self._plts[0,0]

    @property
    def plts(self) -> np.ndarray:
        """Returns all subplots."""
        return self._plts

    def setPlotGrid(
        self,
        rows: int,
        cols: int,
        linkX: bool = False,
        linkY: bool = False,
        aspectLocked: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None
    ):
        """
        Creates multiple subplots in a grid.

        Parameters
        ----------
        rows : int
            Number of rows of subplots.

        cols : int
            Number of columns of subplots.

        linkX : bool
            Whether to globally link all subplots' x-axes.

        linkY : bool
            Whether to globally link all subplots' y-axes.

        aspectLocked : bool
            Whether to globally lock all aspect ratios i.e. 'equal' aspect ratio.

        xlabel : str
            Common x-axis label for all subplots. X-axis labels use the 'bottom' axis.

        ylabel : str
            Common y-axis label for all subplots. Y-axis labels use the 'left' axis.
        """
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
            if xlabel is not None:
                plt.setLabel('bottom', xlabel)
            if ylabel is not None:
                plt.setLabel('left', ylabel)
            # TODO: if both links specified then enable across subplot mouse tracking
            self._plts[i, j] = plt

    def keyPressEvent(self, ev):
        curPlt = self[self._currPlotIndex[0], self._currPlotIndex[1]]
        if ev.key() == Qt.Key.Key_V:
            # Toggle cursor mode between position and value (all plots)
            for plt in np.nditer(self.plts, ['refs_ok']):
                plt.item()._rotateCursorMode() # pyright: ignore
        elif ev.key() == Qt.Key.Key_C:
            # Toggle text colour (all plots)
            for plt in np.nditer(self.plts, ['refs_ok']):
                plt.item()._toggleTextColour() # pyright: ignore
        elif ev.key() == Qt.Key.Key_Escape and self._isMaximized:
            self.subplotMinimize()
        elif ev.key() == Qt.Key.Key_I:
            curPlt._toggleImage()
        elif ev.key() == Qt.Key.Key_L:
            # Toggle magnetized cursor locks (all plots)
            for plt in np.nditer(self.plts, ['refs_ok']):
                plt.item()._toggleLockedPointing() # pyright: ignore
        return super().keyPressEvent(ev)

    def subplotMaximize(self):
        i, j = self._currPlotIndex
        plt = self[i,j].base
        # wipe layout, items are still cached
        self.clear() # pyright:ignore
        self.addItem(plt, row=0, col=0) # pyright: ignore
        # TODO: for any plot other than 0,0 it seems the xlim/ylim is not retained
        self._isMaximized = True
        self.setWindowTitle(self.windowTitle() + f"[{i},{j}]")

    def subplotMinimize(self):
        self.clear() # pyright:ignore
        for (i, j), plt in np.ndenumerate(self._plts):
            self.addItem(plt.base, row=i, col=j) # pyright:ignore
        self._isMaximized = False
        # Reset the window title
        title = self.windowTitle()
        sidx = title.rfind('[')
        self.setWindowTitle(title[:sidx])

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
                    plt.mouseLabel.show()
                    # Update text and position
                    plt._setMouseLabelTextAndPos()
                    # Hide cursors for other plots
                    self._hideInactivePlotCursors()

                    return

    def _hideInactivePlotCursors(self):
        for index, plt in np.ndenumerate(self.plts):
            if not np.all(index == self._currPlotIndex):
                plt.mouseLabel.hide()



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
    f.plt.setAspectLocked()
    f.plt.rectangle((-1,-1), (cols + 1, rows + 1))
    f.show()

    # try custom circles
    ellipseItem = f.plt.ellipses(np.array([[0,0,1,1], [3,3,2,1]]))
    ellipseItem.addEllipse(np.array([1,1,2,3]), pg.mkPen("w"))
    circleItem = f.plt.circles(np.array([[2,2,0.1],[2,3,0.1]]), pg.mkPen("b"))
    circleItem.addEllipse(np.array([4,4,1,1]), pg.mkPen("m"))

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
        f4.setPlotGrid(2,2,True,True,True,xlabel='x',ylabel='y')
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

