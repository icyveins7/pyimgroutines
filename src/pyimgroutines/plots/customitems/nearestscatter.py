from __future__ import annotations
from typing import Any, Callable

import numpy as np
import pyqtgraph as pg
from scipy.spatial import KDTree
from pyqtgraph.graphicsItems.ScatterPlotItem import SpotItem


AnnotationFormatter = Callable[[int, float, float, Any], str]


def defaultAnnotationFormatter(index: int, x: float, y: float, data: Any) -> str:
    """
    Default text for a scatter point annotation.

    Parameters
    ----------
    index : int
        Position of the point in the scatter arrays.
    x, y : float
        Coordinates of the point.
    data : Any
        Per-point payload supplied via ``setData(..., data=...)``; unused here.

    Returns
    -------
    str
        Two-line label with the coordinates and the index.
    """
    return f"X,Y: ({x:.6g}, {y:.6g})\nIndex: {index}"


class NearestScatterPlotItem(pg.ScatterPlotItem):
    """
    A ScatterPlotItem that reports only the nearest point on hover.

    A KDTree is rebuilt whenever the scatter data is changed. Hover queries
    use the nearest point instead of testing every point.

    The item also carries an annotation formatter, a callable that turns a
    point into label text. `PgPlotItem` calls it when the user presses the
    annotation hotkey over a point of this item. Replace it at any time via
    `setAnnotationFormatter`; the callable may look values up lazily
    (e.g. from a database) since it is only invoked on demand.
    """

    def __init__(self, *args, **kwargs):
        self._hoverTree = None
        self._hoverTreeIndices = np.empty(0, dtype=np.intp)
        self._hoverPixelRadii = np.empty(0, dtype=float)
        self._hoverMaxPixelRadius = 0.0
        self._hoverMaxDataRadius = 0.0
        self._hoveredIndices = np.empty(0, dtype=np.intp)
        self._toolTipCleared = True
        self._annotationFormatter: AnnotationFormatter = defaultAnnotationFormatter
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)

    def setAnnotationFormatter(self, formatter: AnnotationFormatter | None):
        """
        Set the callable used to build annotation text for a point.

        Parameters
        ----------
        formatter : callable or None
            ``formatter(index, x, y, data) -> str``. ``index`` is the point's
            position in the scatter arrays, ``x``/``y`` its coordinates and
            ``data`` the per-point payload (None if not supplied). Passing
            None restores `defaultAnnotationFormatter`.
        """
        self._annotationFormatter = (
            defaultAnnotationFormatter if formatter is None else formatter
        )

    def annotationText(self, index: int) -> str:
        """
        Build the annotation text for the point at `index`.

        Parameters
        ----------
        index : int
            Position of the point in the scatter arrays.

        Returns
        -------
        str
            Result of the current annotation formatter.
        """
        # Pull the point's coordinates and payload, then defer to the formatter
        # TODO: baseline pyqtgraph effectively stores as an array of structs,
        # but this is actually inefficient. for now we will maintain it, but
        # early tests show that dict of arrays should be faster
        x = float(self.data["x"][index])
        y = float(self.data["y"][index])
        data = self.data["data"][index]
        return self._annotationFormatter(int(index), x, y, data)

    def nearestPoint(self, x: float, y: float) -> tuple[int, float] | None:
        """
        Return the nearest point whose symbol covers data position (x, y).

        Parameters
        ----------
        x, y : float
            Query position in data coordinates.

        Returns
        -------
        (index, distance) or None
            Scatter index of the nearest point and its distance from the
            query position in data units, or None if the nearest point's
            symbol does not cover the position.
        """
        if self._hoverTree is None:
            return None
        # KDTree nearest neighbour, mapped back to the scatter index
        distance, treeIndex = self._hoverTree.query((x, y))
        index = int(self._hoverTreeIndices[treeIndex])
        # Symbol radius in data units; in pxMode this depends on the zoom
        if self.opts["pxMode"]:
            px, py = self.pixelVectors()
            scale = 0 if px is None or py is None else max(px.length(), py.length())
            radius = self._hoverPixelRadii[index] * scale
        else:
            radius = self.data["size"][index] / 2
        if distance > radius or not self.data["visible"][index]:
            return None
        return index, float(distance)

    def setData(self, *args, **kwargs):
        result = super().setData(*args, **kwargs)

        xy = np.column_stack((self.data["x"], self.data["y"]))
        valid = np.isfinite(xy).all(axis=1)
        self._hoverTreeIndices = np.flatnonzero(valid)
        self._hoverTree = KDTree(xy[valid]) if np.any(valid) else None
        self._hoverPixelRadii = np.maximum(
            self.data["sourceRect"]["w"],
            self.data["sourceRect"]["h"],
        ) / 2
        self._hoverMaxPixelRadius = np.max(self._hoverPixelRadii, initial=0)
        self._hoverMaxDataRadius = np.max(self.data["size"], initial=0) / 2

        return result

    def _spotItem(self, index: int) -> SpotItem:
        item = self.data["item"][index]
        if item is None:
            item = SpotItem(self.data[index], self, index)
            self.data["item"][index] = item
        return item

    def _hoverQueryRadius(self, scale: float = 1) -> float:
        if self._hoverTree is None:
            return 0
        if self.opts["pxMode"]:
            return self._hoverMaxPixelRadius * scale
        return self._hoverMaxDataRadius

    def hoverEvent(self, ev):
        # The original implementation calls self.points()[new] after _maskAt().
        # points() scans all data records to initialize/check SpotItems, even when
        # new contains only one point. Calling super().hoverEvent() would therefore
        # retain an O(N) operation and defeat the purpose of the KD-tree query.
        hasHoverStyle = self._hasHoverStyle()
        self.data["hovered"][self._hoveredIndices] = False
        if hasHoverStyle:
            self.data["sourceRect"][self._hoveredIndices] = 0

        if ev.exit or self._hoverTree is None:
            indices = np.empty(0, dtype=np.intp)
            points = np.empty(0, dtype=object)
        else:
            pos = ev.pos()
            if self.opts["pxMode"]:
                px, py = self.pixelVectors()
                scale = 0 if px is None or py is None else max(px.length(), py.length())
            else:
                scale = 1
            candidateTreeIndices = self._hoverTree.query_ball_point(
                (pos.x(), pos.y()),
                self._hoverQueryRadius(scale),
            )
            indices = self._hoverTreeIndices[candidateTreeIndices]
            distances = np.hypot(
                self.data["x"][indices] - pos.x(),
                self.data["y"][indices] - pos.y(),
            )
            if self.opts["pxMode"]:
                radii = self._hoverPixelRadii[indices] * scale
            else:
                radii = self.data["size"][indices] / 2
            indices = indices[
                (distances <= radii) & self.data["visible"][indices]
            ]
            points = np.array([self._spotItem(i) for i in indices], dtype=object)

        self.data["hovered"][indices] = True
        if hasHoverStyle:
            self.data["sourceRect"][indices] = 0
        self._hoveredIndices = indices

        if len(indices) > 0 and hasHoverStyle:
            self.updateSpots()

        viewBox = self.getViewBox()
        if viewBox is not None and self.opts["tip"] is not None:
            if len(points) > 0:
                cutoff = 3
                tips = [self.opts["tip"](
                    x=point.pos().x(),
                    y=point.pos().y(),
                    data=point.data(),
                ) for point in points[:cutoff]]
                if len(points) > cutoff:
                    tips.append(f"({len(points) - cutoff} others...)")
                viewBox.setToolTip("\n\n".join(tips))
                self._toolTipCleared = False
            elif not self._toolTipCleared:
                viewBox.setToolTip("")
                self._toolTipCleared = True

        self.sigHovered.emit(self, points, ev)
