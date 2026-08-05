from PySide6.QtCore import QObject, Signal
import numpy as np
import pyqtgraph as pg
from ...packed_spatial_grid import PackedSpatialGrid

class RestrictedScatterItem(QObject):
    """
    Scatter plot that only displays raw points when the viewbox is zoomed in
    sufficiently (i.e. the visible tile span is within a threshold).
    At coarser zoom levels the scatter is hidden entirely.
    """
    sigTilesChanged = Signal()

    def __init__(
        self,
        points: np.ndarray,
        tile_size,
        max_tile_span: int = 3,
        symbol="o",
        brush="w",
        name=None,
    ):
        super().__init__()
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if max_tile_span < 1:
            raise ValueError("max_tile_span must be positive")

        self._grid = PackedSpatialGrid(points, tile_size)
        self._scatter = pg.ScatterPlotItem(
            pen=None,
            symbol=symbol,
            brush=brush,
            name=name,
        )
        self._scatter.setZValue(0)
        self._scatter.hide()

        self._max_tile_span = max_tile_span
        self._active_tile_indices = None
        self._showing_raw = False

    @property
    def grid(self) -> PackedSpatialGrid:
        return self._grid

    @property
    def scatter(self) -> pg.ScatterPlotItem:
        return self._scatter

    @property
    def active_tile_indices(self) -> np.ndarray | None:
        return self._active_tile_indices

    @property
    def showing_raw(self) -> bool:
        return self._showing_raw

    @property
    def max_tile_span(self) -> int:
        return self._max_tile_span

    def setRawTiles(self, tile_indices: np.ndarray):
        """Populate the raw scatter item from the selected tile indices."""
        tile_indices = np.asarray(tile_indices)
        if tile_indices.shape != (len(tile_indices), 2):
            raise ValueError("tile_indices must have shape (N, 2)")

        if (
            self._active_tile_indices is not None
            and np.array_equal(tile_indices, self._active_tile_indices)
        ):
            return

        tile_points = [
            self._grid.getTilePoints(tile_idx)
            for tile_idx in tile_indices
        ]
        if tile_points:
            points = np.concatenate(tile_points, axis=0)
        else:
            points = np.empty((0, 2), dtype=self._grid.point_buffer.dtype)

        # print(f"RestrictedScatterItem.setRawTiles: {tile_indices}, total {len(points)}")
        self._scatter.setData(
            x=points[:, 0],
            y=points[:, 1],
        )
        self._active_tile_indices = tile_indices.copy()
        self.sigTilesChanged.emit()

    def showRaw(self):
        if not self._showing_raw:
            print("RestrictedScatterItem.showRaw")
        self._scatter.show()
        self._showing_raw = True

    def hideRaw(self):
        if self._showing_raw:
            print("RestrictedScatterItem.hideRaw")
        self._scatter.hide()
        self._showing_raw = False
