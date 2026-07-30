import numpy as np
import pyqtgraph as pg
from scipy.spatial import KDTree
from pyqtgraph.graphicsItems.ScatterPlotItem import SpotItem


class NearestScatterPlotItem(pg.ScatterPlotItem):
    """
    A ScatterPlotItem that reports only the nearest point on hover.

    A KDTree is rebuilt whenever the scatter data is changed. Hover queries
    use the nearest point instead of testing every point.
    """

    def __init__(self, *args, **kwargs):
        self._hoverTree = None
        self._hoverTreeIndices = np.empty(0, dtype=np.intp)
        self._hoverPixelRadii = np.empty(0, dtype=float)
        self._hoveredIndex = None
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)

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

        return result

    def _hoverRadius(self, index: int) -> float:
        if self.opts["pxMode"] and self.opts["useCache"]:
            size = self._hoverPixelRadii[index]
        else:
            size = self.data["size"][index] / 2

        if not self.opts["pxMode"]:
            return size

        px, py = self.pixelVectors()
        if px is None or py is None:
            return 0
        return size * max(px.length(), py.length())

    def _spotItem(self, index: int) -> SpotItem:
        item = self.data["item"][index]
        if item is None:
            item = SpotItem(self.data[index], self, index)
            self.data["item"][index] = item
        return item

    def hoverEvent(self, ev):
        # The original implementation calls self.points()[new] after _maskAt().
        # points() scans all data records to initialize/check SpotItems, even when
        # new contains only one point. Calling super().hoverEvent() would therefore
        # retain an O(N) operation and defeat the purpose of the KD-tree query.
        changed = []
        if self._hoveredIndex is not None:
            changed.append(self._hoveredIndex)
            self.data["hovered"][self._hoveredIndex] = False
            self.data["sourceRect"][self._hoveredIndex] = 0
            self._hoveredIndex = None

        if ev.exit or self._hoverTree is None:
            points = np.empty(0, dtype=object)
        else:
            pos = ev.pos()
            distance, treeIndex = self._hoverTree.query((pos.x(), pos.y()))
            index = self._hoverTreeIndices[treeIndex]
            radius = self._hoverRadius(index)

            if distance <= radius and self.data["visible"][index]:
                self.data["hovered"][index] = True
                self.data["sourceRect"][index] = 0
                self._hoveredIndex = index
                changed.append(index)
                points = np.array([self._spotItem(index)], dtype=object)
            else:
                points = np.empty(0, dtype=object)

        if changed and self._hasHoverStyle():
            self.updateSpots()

        viewBox = self.getViewBox()
        if viewBox is not None and self.opts["tip"] is not None:
            if len(points) > 0:
                point = points[0]
                viewBox.setToolTip(self.opts["tip"](
                    x=point.pos().x(),
                    y=point.pos().y(),
                    data=point.data(),
                ))
                self._toolTipCleared = False
            elif not self._toolTipCleared:
                viewBox.setToolTip("")
                self._toolTipCleared = True

        self.sigHovered.emit(self, points, ev)
