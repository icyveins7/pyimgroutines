import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Signal, QRectF

from ...packed_spatial_grid import PackedSpatialGrid
from ...point_raster import points_to_image


class HybridScatterItem(QObject):
    sigTilesChanged = Signal()

    """
    Controller for a coarse image and a tile-backed raw scatter plot.
    """

    def __init__(
        self,
        points: np.ndarray,
        tile_size,
        img_dims=(2048, 2048),
        img_xywh=None,
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
        coarse_image, self._coarse_xywh = points_to_image(
            points,
            img_dims=img_dims,
            img_xywh=img_xywh,
        )
        self._coarseimg = pg.ImageItem(coarse_image, axisOrder="row-major")
        self._coarseimg.setRect(QRectF(*self._coarse_xywh))
        self._coarseimg.setZValue(-100)

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
        self._showing_coarse = True

    @property
    def grid(self) -> PackedSpatialGrid:
        return self._grid

    @property
    def coarseimg(self) -> pg.ImageItem:
        return self._coarseimg

    @property
    def scatter(self) -> pg.ScatterPlotItem:
        return self._scatter

    @property
    def active_tile_indices(self) -> np.ndarray | None:
        return self._active_tile_indices

    @property
    def showing_coarse(self) -> bool:
        return self._showing_coarse

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

        print(f"HybridScatterItem.setRawTiles: {tile_indices}, total {len(points)}")
        self._scatter.setData(
            x=points[:, 0],
            y=points[:, 1],
        )
        self._active_tile_indices = tile_indices.copy()
        self.sigTilesChanged.emit()

    def showCoarse(self):
        if not self._showing_coarse:
            print("HybridScatterItem.showCoarse")
        self._coarseimg.show()
        self._scatter.hide()
        self._showing_coarse = True

    def showRaw(self):
        if self._showing_coarse:
            print("HybridScatterItem.showRaw")
        self._coarseimg.hide()
        self._scatter.show()
        self._showing_coarse = False
