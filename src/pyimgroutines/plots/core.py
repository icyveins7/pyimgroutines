from __future__ import annotations
from typing import Iterable, Protocol
from PySide6.QtGui import QBrush, QPen, QTextBlockFormat, QTextCursor, QColor
import pyqtgraph as pg
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import QPointF, Qt, QRectF, Signal, QObject
from PySide6.QtWidgets import QApplication, QMessageBox, QMainWindow
from itertools import repeat

from ._keybuffer import KeyBufferCoordinates
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

class PgPlotItem(QObject):
    """
    A wrapper around default PlotItem with extra references.
    This is primarily to aid in image-related functionality.

    This is a subclass of QObject in order to enable signal/slot connections.
    """
    trackingDtype = np.float64

    # Constants used for cursor tracking modes
    CURSOR_SHOW_NONE = 0
    CURSOR_SHOW_POS = 1
    CURSOR_SHOW_VALUE = 2

    # Constants used for target crosshair modes
    TARGET_NONE = 0
    TARGET_LIGHT = 1
    TARGET_DARK = 2

    # Custom signals
    sigROIselectionChangeFinished = Signal(np.ndarray)
    sigMaskChanged = Signal(np.ndarray)

    def __init__(self, plotItem: pg.PlotItem, parent: PgFigure):
        super().__init__() # for signal/slot
        self._plotItem = plotItem
        self._parent = parent
        # Extras
        self._cursorMode = self.CURSOR_SHOW_POS
        self._im = None
        # NOTE: where the cursor actually is, not necessarily where the label is
        self._cursorPos = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._btmLeftPos = np.array([np.nan, np.nan], dtype=self.trackingDtype)
        self._pixelSize = np.array([np.nan, np.nan], dtype=self.trackingDtype)

        self._mouseLabel = pg.TextItem(text="", anchor=(0, 1))
        self._toggleTextColour() # Just run it so that we get our background colour
        self.addItem(self._mouseLabel, ignoreBounds=True) # On at the start
        # Set custom slot for mouseMoved/mouseClicked
        self._parent.scene().sigMouseMoved.connect(self._parent.mouseMoved) # pyright: ignore
        self._parent.scene().sigMouseClicked.connect(self._parent.mouseClicked) # pyright: ignore

        self._imgData = np.empty((0, 0), dtype=np.float32)
        self._cbar = pg.ColorBarItem()
        self._lockedPointing = False
        self._addHalfPixelBorder = False
        self._roi = pg.ROI((0, 0),
                           movable=True,
                           rotatable=False,
                           resizable=True,
                           pen=pg.mkPen('k'),
                           hoverPen=pg.mkPen((150,150,150))) # created but not added
        self._roi.addScaleHandle((1, 0),(0, 1))
        self._roi.addScaleHandle((0, 0),(1, 1))
        self._roi.addScaleHandle((0, 1),(1, 0))
        self._roi.addScaleHandle((1, 1),(0, 0))
        self._roi.sigRegionChangeFinished.connect(self.onROIchangeFinished)
        self._mask = np.zeros((1, 1), dtype=bool)
        self._minimap = pg.ImageItem()
        self._target = pg.TargetItem(movable=False)
        self._targetMode = self.TARGET_NONE

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
    def target(self) -> pg.TargetItem:
        return self._target

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def roi(self) -> pg.ROI:
        return self._roi

    @property
    def im(self) -> pg.ImageItem | None:
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

    def zoomTo(
        self,
        pos: tuple[tuple[float,float]|float|None, tuple[float,float]|float|None]
    ):
        # TODO: add docstring
        # NOTE: with aspectLocked, this automatically scales to maintain
        # equal aspect ratio, so it may not exactly respect ranges
        # TODO: ideally for both x/y should support
        # None -> use old range exactly
        # A -> centre around A, maintain old span i.e. A +/- span/2
        # A:B -> directly set range using this
        # A: -> set range from A to upper limit of image bbox
        # :B -> set range from lower limit of image bbox to B
        # : -> exactly enclose image bbox for this coordinate

        # Get the current span
        viewRange = self.viewRange()
        finalRanges = list()

        for p, curRange in zip(pos, viewRange):
            # If the type is a tuple, we use it directly
            if isinstance(p, tuple):
                finalRanges.append(p)
            # If type is float, we maintain the current span
            elif isinstance(p, float):
                span = curRange[1] - curRange[0]
                finalRanges.append((p-span/2, p+span/2))
            # If None, don't change current range
            elif p is None:
                finalRanges.append(curRange)

        # print(finalRanges)
        self.setRange(xRange=finalRanges[0],
                      yRange=finalRanges[1],
                      padding=0)

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
        xMesh_yMesh: tuple[np.ndarray, np.ndarray] | None = None,
        levels: tuple[float, float] | list | None = None,
        addHalfPixelBorder: bool = True,
        zvalue: int = -100,
        cmap: pg.colormap.ColorMap | None = pg.colormap.get("viridis"),
        colorbar: bool = True,
        includeLegend: bool = False,
        addMinimap: bool = False # TODO: default True when actually ready
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
            This is the prioritised method to define the bounding rectangle;
            see xMesh_yMesh for the alternative method if this is None.
            If both methods are left None, it is equivalent to setting this to
            (0, 0) and the number of pixels respectively.

        xMesh_yMesh : tuple[np.ndarray, np.ndarray]
            This alternative method of specifying the bounding rectangle is
            a convenient argument when the x and y meshgrids are available.
            This function will automatically calculate the necessary xywh argument
            above.

        levels : tuple[float, float] | list | None
            Sets levels on the ImageItem (see .setLevels()).
            Defaults to None, which will automatically compute min/max value of the image,
            same as the default for pg.ImageItem if no levels are set.

        addHalfPixelBorder : bool
            Whether to automatically add half a pixel as a border around the entire image.
            This ensures that pixels are centred on the points, rather than having the pixels
            be vertices.

            Example: a 2x2 matrix will extend from -0.5 to +1.5 pixels instead of 0 to 2.

            Defaults to True.

        zvalue : int
            Priority/height of the image. Lower numbers will paint the image 'behind' other items.

        cmap : pg.colormap.ColorMap | None
            Colormap to use for the image.
            Defaults to viridis. Use None if you want grayscale,
            which is pyqtgraph's default.

        colorbar : bool
            Whether to include a colorbar. Defaults to True.

        includeLegend : bool
            Whether to include a legend. The image item itself is not added to the legend,
            but this is useful so other traditional plot items can be added and show up.
        """
        # TODO: things are not handled well when multiple images are added
        if includeLegend:
            self.addLegend()

        # Save reference to image data for this subplot
        self._imgData = arr
        if levels is None:
            # Have to manually get min/max again since setting
            # levels to None specifically disables the autoLevels
            levels = (np.nanmin(arr), np.nanmax(arr))
        self._im = pg.ImageItem(arr, axisOrder='row-major', levels=levels) # default to row-major instead

        pixelWidth = 1
        pixelHeight = 1

        # Default to 0,0 bottom left, width and height in pixels
        if xywh is None:
            if xMesh_yMesh is None:
                xywh = [0, 0, arr.shape[1], arr.shape[0]]
            else:
                # Calculate from the meshes
                xMesh, yMesh = xMesh_yMesh
                xUniqueMesh = np.sort(np.unique(xMesh))
                yUniqueMesh = np.sort(np.unique(yMesh))
                pixelWidth = xUniqueMesh[1] - xUniqueMesh[0]
                pixelHeight = yUniqueMesh[1] - yUniqueMesh[0]
                xywh = [
                    np.min(xMesh),
                    np.min(yMesh),
                    np.max(xMesh) - np.min(xMesh) + pixelWidth,
                    np.max(yMesh) - np.min(yMesh) + pixelHeight
                ]

        # Centres each pixel on the grid coordinates (instead of corners)
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
        self._im.setAutoDownsample(False)

        # cm2use = pg.colormap.getFromMatplotlib("viridis")
        if cmap is not None:
            self._im.setLookupTable(cmap.getLookupTable()) # pyright: ignore

        if colorbar:
            self._cbar = self.addColorBar(
                self._im,
                colorMap=cmap,
                interactive=False # default false because large images will lag
            )

        self.addItem(self._im)

        # TODO: DEPRECATED: textitem previously set here but now in ctor
        # Slot to show cursor position
        # self._mouseLabel = pg.TextItem(text="", anchor=(0, 1)) # show to top right of cursor
        # self._mouseLabel.setFlag(self._mouseLabel.GraphicsItemFlag.ItemIgnoresTransformations)
        # self.addItem(self._mouseLabel, ignoreBounds=True)
        # # Set custom slot for mouseMoved/mouseClicked
        # self._parent.scene().sigMouseMoved.connect(self._parent.mouseMoved) # pyright: ignore
        # self._parent.scene().sigMouseClicked.connect(self._parent.mouseClicked) # pyright: ignore

        # Disable auto-range, tends to be buggy/annoying/enter loops
        self.disableAutoRange(pg.ViewBox.XYAxes)

        if addMinimap:
            # TODO: need to draw the rect showing current FOV
            targetLength = 64
            targetDsrH = self._imgData.shape[0] // targetLength
            targetDsrW = self._imgData.shape[1] // targetLength
            targetDsrH = 1 if targetDsrH == 0 else targetDsrH
            targetDsrW = 1 if targetDsrW == 0 else targetDsrW
            self._minimap.setImage(
                self._imgData[::targetDsrH, ::targetDsrW],
                axisOrder='row-major'
            )
            self.scene().addItem(self._minimap)
            # Make sure to position it relative to the subplot
            self.repositionMinimap()

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
        # Invert the background
        cur = QTextCursor(self._mouseLabel.textItem.document())
        cur.select(QTextCursor.SelectionType.BlockUnderCursor)
        bf = QTextBlockFormat()
        bf.setBackground(QColor(colour.red(), colour.green(), colour.blue(), 128))
        cur.mergeBlockFormat(bf)

    def _getNearestImagePointIndex(self, pos: np.ndarray | None = None) -> np.ndarray | None:
        if pos is None:
            pos = self._cursorPos
        # NOTE: cursorPos may be nan/invalid if hovering over another subplot
        if np.all(np.isnan(pos)):
            return None
        offset = pos - self._btmLeftPos
        index = offset / self._pixelSize # this is in x/y
        dataRows, dataCols = self._imgData.shape
        if np.any(index < 0) or index[0] > dataCols or index[1] > dataRows: # pyright: ignore
            return None
        return index.astype(np.int32)

    def _getLockedPosition(self):
        index = self._getNearestImagePointIndex()
        if index is None:
            return None, None
        if self._addHalfPixelBorder:
            bottomLeftPointerPos = self._btmLeftPos + self._pixelSize * 0.5
        else:
            bottomLeftPointerPos = self._btmLeftPos
        return index * self._pixelSize + bottomLeftPointerPos, index

    def _replaceMouseLabelText(self, newtext: str):
        # The pyqtgraph one will reset all formatting;
        # we have to use qtextcursor directly
        c = QTextCursor(self._mouseLabel.textItem.document())
        c.select(QTextCursor.SelectionType.Document)
        c.insertText(newtext)

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
                self._replaceMouseLabelText(f"{pos[0]:.6g}, {pos[1]:.6g}")
            elif self._cursorMode == self.CURSOR_SHOW_VALUE:
                index = self._getNearestImagePointIndex() # pyright: ignore
                if index is None:
                    self._replaceMouseLabelText(f"OOB") # pyright: ignore
                else:
                    self._replaceMouseLabelText(f"[{int(index[1])}, {int(index[0])}]: {self._imgData[int(index[1]), int(index[0])]}") # pyright: ignore
            else:
                self._replaceMouseLabelText("")
        else:
            self._replaceMouseLabelText("")


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

    def _toggleROI(self):
        if self._roi in self.base.items:
            self.removeItem(self._roi)
        else:
            viewrangeX, viewrangeY = self.viewRange()
            centre = np.array([
                0.5*(viewrangeX[0]+viewrangeX[1]), 0.5*(viewrangeY[0]+viewrangeY[1])
            ])
            wh = np.array([
                (viewrangeX[1]-viewrangeX[0])/2, (viewrangeY[1]-viewrangeY[0])/2])
            self._roi.setPos(centre-wh/2)
            self._roi.setSize(wh)
            self.addItem(self._roi)
        self.onROIchangeFinished(self._roi) # trigger explicitly for both show/hide

    def onROIchangeFinished(self, roi: pg.ROI):
        if self.im is None:
            return

        roiPos = roi.pos()
        roiSize = roi.size()
        # print(roiPos)
        # print(roiSize)
        # Note that this is X then Y
        startImgIdx = self._getNearestImagePointIndex(np.array(roiPos))
        endImgIdx = self._getNearestImagePointIndex(
                np.array(roiPos + roiSize)) + 1 # add 1 to include the final
        # TODO: handle for corners outside bounds

        # print(startImgIdx)
        # print(endImgIdx)

        # Only send selection if the ROI is actually active
        if self._roi not in self.base.items:
            selection = np.array([])
        else:
            selection = self._imgData[startImgIdx[1]:endImgIdx[1], startImgIdx[0]:endImgIdx[0]]
        self.sigROIselectionChangeFinished.emit(selection)

    def setMaskFromLinearRegionItem(self, item: pg.LinearRegionItem):
        # TODO: maybe make a custom MaskItem
        if self._imgData is None:
            return

        region = item.getRegion()
        self._mask = np.logical_and(
            self._imgData >= region[0],
            self._imgData <= region[1]
        )
        self.sigMaskChanged.emit(self._mask)

    def linkToLinearRegionItem(self, item: pg.LinearRegionItem):
        # TODO: maybe make a custom MaskItem
        item.sigRegionChangeFinished.connect(self.setMaskFromLinearRegionItem)

    def _toggleMinimap(self):
        if self._minimap in self.scene().items():
            self.scene().removeItem(self._minimap)
        else:
            self.scene().addItem(self._minimap)

    def repositionMinimap(self):
        pltWindowPos = self.base.pos()
        self._minimap.setRect(pltWindowPos.x()+50,pltWindowPos.y()+50,32,-32) # set height negative so it's +ve y upwards

    def _handleTargeting(self):
        if self._lockedPointing:
            lockedPos, _ = self._getLockedPosition()
            if lockedPos is not None:
                self._target.setPos(lockedPos[0], lockedPos[1])
        else:
            self._target.setPos(self._cursorPos) # could also use self._cursorPos?

    def _toggleTargeting(self):
        self._targetMode = (self._targetMode + 1) % 3
        if self._targetMode == self.TARGET_NONE:
            self.removeItem(self._target)
        else:
            if self._target not in self.base.items:
                self.addItem(self._target)
            if self._targetMode == self.TARGET_LIGHT:
                self._target.setPen(pg.mkPen(255,255,0))
            if self._targetMode == self.TARGET_DARK:
                self._target.setPen(pg.mkPen(0,0,255))

class PgFigure(QMainWindow):
    """
    A pyqtgraph 'figure', built on a QMainWindow containing a GraphicsLayoutWidget,
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
        - Status bar for displaying information
    """
    FIGURE_INDEX = 1

    def __init__(self, *args, **kwargs):
        """
        Instantiate a new PgFigure, with a default single subplot.

        All input arguments are passed to pg.GraphicsLayoutWidget().
        """
        # Ensure QApplication exists before creating any widgets
        # TODO: check if there's a better way rather than always making a QApp?
        pg.mkQApp()

        self._figIndex = self._getFigureIndex()
        title = kwargs.pop("title", f"PgFigure_{self._figIndex}")
        super().__init__()
        self.setWindowTitle(title)

        # Create the graphics widget and set as central widget
        self._graphicsWidget = pg.GraphicsLayoutWidget(*args, **kwargs)
        self._graphicsWidget.setAntialiasing(False)
        self.setCentralWidget(self._graphicsWidget)

        self._plts = np.empty((0, 0), dtype=PgPlotItem)
        self._currPlotIndex = np.array([0, 0], dtype=np.uint8)
        self.setPlotGrid(1, 1) # Default to having a single plot

        self._isMaximized = False

        self._keybuffer = KeyBufferCoordinates()
        self._keybuffer.bufferChanged.connect(self._onKeyBufferChanged)

    # Forward unknown attributes to the graphics widget
    def __getattr__(self, name):
        # Avoid infinite recursion during init
        if '_graphicsWidget' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._graphicsWidget, name)

    @property
    def graphicsWidget(self) -> pg.GraphicsLayoutWidget:
        """Return the underlying GraphicsLayoutWidget."""
        return self._graphicsWidget

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
        """
        Returns all subplots.
        To do a simple iteration over all of them, use np.nditer(..., ['refs_ok'])
        and then use .item() to get the object (the PgPlotItem).
        """
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
            # Disable ViewBox key handling, go straight to our custom handling
            # This disables the Ctrl -/+ zoom and Ctrl A reset, but that's only used
            # in mouseRect mode anyway which i don't intend to target
            plt.vb.keyPressEvent = lambda ev: self.keyPressEvent(ev)
            # TODO: if both links specified then enable across subplot mouse tracking
            self._plts[i, j] = plt

    def keyPressEvent(self, ev):
        curPlt = self[self._currPlotIndex[0], self._currPlotIndex[1]]
        key = self._keybuffer.parseKey(ev.key())
        if key is None:
            return

        # Check if buffer is frozen first (G-commands)
        if self._keybuffer.frozen:
            # GG: go/zoom to coordinates
            if ev.key() == Qt.Key.Key_G:
                coords = self._keybuffer.flushCoordinates()
                curPlt.zoomTo(coords)
            # GC: change the colorbar range
            elif ev.key() == Qt.Key.Key_C:
                lower, upper = self._keybuffer.flushRange()
                origLower, origUpper = curPlt.cbar.levels()
                lower = origLower if lower is None else lower
                upper = origUpper if upper is None else upper
                curPlt.cbar.setLevels((lower, upper))
            # Always unfreeze at the end
            self._keybuffer.unfreeze()
        # Normal key handling (not frozen)
        elif ev.key() == Qt.Key.Key_G:
            # First G: freeze the buffer, wait for next key
            self._keybuffer.freeze()
        elif ev.key() == Qt.Key.Key_V:
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
        elif ev.key() == Qt.Key.Key_H:
            helpbox = self._makeHelpDialog()
            helpbox.exec()
        elif ev.key() == Qt.Key.Key_B:
            # Toggle status bar visibility
            sb = self.statusBar()
            sb.setVisible(not sb.isVisible())
        elif ev.key() == Qt.Key.Key_R:
            # Toggle ROI
            curPlt._toggleROI() # pyright: ignore
        elif ev.key() == Qt.Key.Key_M:
            # Toggle minimap
            curPlt._toggleMinimap() # pyright: ignore
        elif ev.key() == Qt.Key.Key_A:
            # Toggle aspect ratio locking
            isLocked = curPlt.vb.getState()['aspectLocked']
            curPlt.setAspectLocked(not isLocked)
        elif ev.key() == Qt.Key.Key_T:
            # Toggle targeting for all plots
            for plt in np.nditer(self.plts, ['refs_ok']):
                plt.item()._toggleTargeting() # pyright: ignore
        elif ev.key() == Qt.Key.Key_Shift:
            # Mirror cursor and target
            self._mirrorCursorAndTarget(True)
        else:
            # Key not handled by us, let Qt propagate it
            return super().keyPressEvent(ev)

        # If we handled the key, accept it to prevent double-firing
        ev.accept()

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Shift:
            # Mirror cursor and target
            self._mirrorCursorAndTarget(False)
        else:
            # Key not handled by us, let Qt propagate it
            return super().keyReleaseEvent(ev)

    def _makeHelpDialog(self):
        helpbox = QMessageBox()
        helpbox.setWindowTitle("Hotkeys")
        helpText = """
h: Show this help window
v: Rotate cursor's text modes (position / data / none)
c: Rotate cursor's text colours
l: Toggle magnetized cursor locks
b: Toggle status bar
a: Toggle aspect ratio lock
r: Toggle ROI
t: Toggle targeting crosshair (will follow current magnetization)
"""
        helpbox.setText(helpText)
        return helpbox

    def _mirrorCursorAndTarget(self, enabled: bool):
        # TODO: take current plot and apply the same cursor/target
        # positions to other plots, along with toggling visibility
        # print(f"_mirrorCursorAndTarget {enabled}")
        pass

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
        curPlt = None
        # NOTE: when maximized, it seems like the sceneBoundingRect
        # is inaccurate, so we force it
        if self._isMaximized:
            curPlt = self._plts[self._currPlotIndex[0], self._currPlotIndex[1]]
        else:
            found = False
            for i in range(self._plts.shape[0]):
                for j in range(self._plts.shape[1]):
                    plt = self._plts[i,j]
                    if plt.im is None:
                        continue

                    # TODO: eventually allow mouse cursor updates only for coordinates
                    # if no image is present?
                    if plt.sceneBoundingRect().contains(evt): # pyright: ignore
                        # Cache that this is the currently hovered plot
                        self._currPlotIndex[:] = [i, j]
                        curPlt = plt
                        # print(f"in {i},{j}")
                        found = True
                        break
                if found:
                    break

        if curPlt is not None:
            coords = curPlt.vb.mapSceneToView(evt) # pyright: ignore
            # Cache internal cursor position
            curPlt._setCursorPositionInPlot(coords) # pyright: ignore
            curPlt.mouseLabel.show() # pyright: ignore
            # Update text and position
            curPlt._setMouseLabelTextAndPos() # pyright: ignore
            # Hide cursors for other plots
            self._hideInactivePlotCursors()

            # Handle targeting
            curPlt.target.show() # pyright: ignore
            curPlt._handleTargeting() # pyright: ignore
            # Hide targets for other plots
            self._hideInactivePlotTargets()

    def _hideInactivePlotCursors(self):
        for index, plt in np.ndenumerate(self.plts):
            # Match everything except current plot
            if not np.array_equal(index, self._currPlotIndex):
                plt.mouseLabel.hide()

    def _hideInactivePlotTargets(self):
        for index, plt in np.ndenumerate(self.plts):
            # Match everything except current plot
            if not np.array_equal(index, self._currPlotIndex):
                plt.target.hide()

    def resizeEvent(self, evt):
        # TODO: handle the custom resizing in the PgPlotItems
        for plt in np.nditer(self.plts, flags=['refs_ok']):
            plt.item().repositionMinimap()
        super().resizeEvent(evt)

    def _onKeyBufferChanged(self):
        s = self._keybuffer.string
        self.statusBar().showMessage(s)


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
        length = 7
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
        f2.plt.image(y, cmap=None)
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
            # print(plt.vb.viewPos())
            print(plt.pos())
            # print(plt.vb.boundingRect())

    forceShow()

