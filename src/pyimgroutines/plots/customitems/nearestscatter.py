import time
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
        self._hoverMaxPixelRadius = 0.0
        self._hoverMaxDataRadius = 0.0
        self._hoveredIndices = np.empty(0, dtype=np.intp)
        self._searchCount = 0
        self._searchTotal = 0.0
        self._paintCount = 0
        self._paintTotal = 0.0
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)

    def paint(self, *args, **kwargs):
        start = time.perf_counter()
        result = super().paint(*args, **kwargs)
        self._paintCount += 1
        self._paintTotal += time.perf_counter() - start
        if self._paintCount <= 3 or self._paintCount % 10 == 0:
            print(
                f"NearestScatterPlotItem paint #{self._paintCount}: "
                f"{(time.perf_counter() - start) * 1000:.3f} ms"
            )
        return result

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

    def _hoverRadius(self, index: int, scale: float = 1) -> float:
        if self.opts["pxMode"] and self.opts["useCache"]:
            size = self._hoverPixelRadii[index]
        else:
            size = self.data["size"][index] / 2

        if not self.opts["pxMode"]:
            return size
        return size * scale

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
            searchStart = time.perf_counter()
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
            self._searchCount += 1
            self._searchTotal += time.perf_counter() - searchStart
            if self._searchCount <= 3 or self._searchCount % 10 == 0:
                print(
                    f"NearestScatterPlotItem search #{self._searchCount}: "
                    f"{(time.perf_counter() - searchStart) * 1000:.3f} ms"
                )
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
