import numpy as np


class PackedSpatialGrid:
    def __init__(self, points: np.ndarray, tile_size: np.ndarray | tuple[float, float]):
        """
        Build a packed spatial grid from two-dimensional point coordinates.

        Parameters
        ----------
        points : array-like, shape (N, 2)
            Point coordinates as ``(x, y)`` pairs. The input dtype is preserved
            in the packed point buffer.

        tile_size : scalar or array-like, shape (2,)
            Tile dimensions as ``(width, height)``. A scalar creates square
            tiles with the same width and height.

        Notes
        -----
        The grid origin is the minimum ``(x, y)`` coordinate in ``points``.
        Points are sorted into contiguous tile ranges during construction, and
        the LUT stores each tile's packed-buffer offset and point count.
        """
        points = np.asarray(points)
        self._tile_size = np.asarray(tile_size)
        if self._tile_size.ndim == 0:
            self._tile_size = np.repeat(self._tile_size, 2)
        if self._tile_size.shape != (2,) or np.any(self._tile_size <= 0):
            raise ValueError("tile_size must be a positive scalar or have shape (2,)")

        # Determine grid dimensions based on point bounds
        self._min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Number of tiles needed to cover the extent of the points
        num_tiles_x = int(np.ceil((max_coords[0] - self._min_coords[0]) / self._tile_size[0])) + 1
        num_tiles_y = int(np.ceil((max_coords[1] - self._min_coords[1]) / self._tile_size[1])) + 1

        self._num_tiles_x = num_tiles_x
        self._num_tiles_y = num_tiles_y
        self._total_tiles = num_tiles_x * num_tiles_y

        # 1. Calculate tile indices for every point
        # We normalize coordinates so the minimum point starts at (0,0)
        normalized_points = points - self._min_coords
        x_indices = (normalized_points[:, 0] // self._tile_size[0]).astype(int)
        y_indices = (normalized_points[:, 1] // self._tile_size[1]).astype(int)

        # Retain each point's 2D tile index as (x_index, y_index).
        self._point_tile_indices = np.column_stack((x_indices, y_indices))

        # Flattened IDs are used internally only to sort the packed buffer.
        tile_ids = x_indices + (y_indices * num_tiles_x)

        # 2. Sort points by their tile ID to create the packed buffer
        sorted_indices = np.argsort(tile_ids)
        self._sort_order = sorted_indices
        self._point_buffer = points[sorted_indices]
        self._point_tile_indices = self._point_tile_indices[sorted_indices]
        sorted_tile_ids = tile_ids[sorted_indices]

        # 3. Build the 2D Look-Up Table (LUT)
        # LUT[y_index, x_index] stores [start_offset, count].
        self._lut = np.zeros((num_tiles_y, num_tiles_x, 2), dtype=np.int32)

        # Find unique tile indices and their counts in the sorted list
        unique_ids, counts = np.unique(sorted_tile_ids, return_counts=True)

        # Calculate cumulative start offsets
        offsets = np.zeros(len(unique_ids), dtype=np.int32)
        offsets[1:] = np.cumsum(counts)[:-1]

        # Populate the LUT. Tile indices are stored as (x, y), while NumPy
        # arrays are indexed as [y, x].
        unique_x = unique_ids % num_tiles_x
        unique_y = unique_ids // num_tiles_x
        self._lut[unique_y, unique_x] = np.column_stack((offsets, counts))

    @property
    def num_tiles_x(self):
        return self._num_tiles_x

    @property
    def num_tiles_y(self):
        return self._num_tiles_y

    @property
    def lut(self):
        return self._lut

    @property
    def point_buffer(self):
        return self._point_buffer

    @property
    def max_tile_count(self):
        """
        Return the maximum number of points in any tile.
        Helps decide if this tile size is sufficient for the use-case
        (or if it should be shrunk further).
        """
        return np.max(self._lut[:, :, 1])

    def getTileIndex(self, x: float, y: float) -> np.ndarray | None:
        """
        Return the (x, y) tile index containing a coordinate.

        Returns None when the coordinate lies outside the grid bounds.
        """
        # Normalize input coordinate using the cached grid origin.
        norm_x = x - self._min_coords[0]
        norm_y = y - self._min_coords[1]

        x_idx = int(norm_x // self._tile_size[0])
        y_idx = int(norm_y // self._tile_size[1])

        if x_idx >= self._num_tiles_x or y_idx >= self._num_tiles_y or x_idx < 0 or y_idx < 0:
            return None

        return np.array([x_idx, y_idx], dtype=np.intp)

    def getTilePoints(self, tileIdx: np.ndarray | None) -> np.ndarray | None:
        """
        Return the points belonging to an (x, y) tile index.
        See getTileIndex().
        """
        if tileIdx is None:
            return None

        tileIdx = np.asarray(tileIdx)
        if tileIdx.shape != (2,):
            raise ValueError("tileIdx must have shape (2,)")

        x_idx, y_idx = tileIdx
        if x_idx < 0 or x_idx >= self._num_tiles_x or y_idx < 0 or y_idx >= self._num_tiles_y:
            raise IndexError(f"Invalid tile index: {tileIdx}")

        start_offset, count = self._lut[y_idx, x_idx]
        return self._point_buffer[start_offset : start_offset + count]

    @property
    def sort_order(self) -> np.ndarray:
        """
        Return the permutation that maps original point indices to their
        packed-buffer order.  If ``v`` is a vector tied to the points,
        use ``v[self.sort_order]`` to rearrange it identically.
        """
        return self._sort_order.copy()

    def getTileCount(self, tileIdx: np.ndarray | None) -> int | None:
        """
        Return the number of points belonging to an (x, y) tile index.
        """
        if tileIdx is None:
            return None

        tileIdx = np.asarray(tileIdx)
        if tileIdx.shape != (2,):
            raise ValueError("tileIdx must have shape (2,)")

        x_idx, y_idx = tileIdx
        if x_idx < 0 or x_idx >= self._num_tiles_x or y_idx < 0 or y_idx >= self._num_tiles_y:
            raise IndexError(f"Invalid tile index: {tileIdx}")

        return int(self._lut[y_idx, x_idx, 1])

    def getTileIndicesOverlappingBox(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float
    ) -> np.ndarray:
        """
        Return all (x, y) tile indices overlapping a bounding box.

        The box is specified by its bottom-left point (x0, y0) and top-right
        point (x1, y1). Tiles with no points are included if they overlap the
        box geometrically.
        """
        if x1 < x0 or y1 < y0:
            raise ValueError("box top-right must not be below or left of bottom-left")

        x_start = int(np.floor((x0 - self._min_coords[0]) / self._tile_size[0]))
        y_start = int(np.floor((y0 - self._min_coords[1]) / self._tile_size[1]))
        x_end = int(np.floor((x1 - self._min_coords[0]) / self._tile_size[0]))
        y_end = int(np.floor((y1 - self._min_coords[1]) / self._tile_size[1]))

        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
        x_end = min(x_end, self._num_tiles_x - 1)
        y_end = min(y_end, self._num_tiles_y - 1)

        if x_start > x_end or y_start > y_end:
            return np.empty((0, 2), dtype=np.intp)

        return np.array([
            [x_idx, y_idx]
            for y_idx in range(y_start, y_end + 1)
            for x_idx in range(x_start, x_end + 1)
        ], dtype=np.intp)


# --- Smoke Test ---

if __name__ == "__main__":
    # Tile (0,0): 1 point
    # Tile (1,0): 2 points
    # Tile (0,1): 2 points
    # Tile (1,1): 1 point
    # Deliberately mixed up here
    raw_points = np.array([
        [10, 10],   # Tile (0,0) - Index 0
        [70, 10],   # Tile (1,0) - Index 1
        [10, 60],   # Tile (0,1) - Index 2
        [60, 10],   # Tile (1,0) - Index 1
        [80, 80],   # Tile (1,1) - Index 3
        [10, 70],   # Tile (0,1) - Index 2
    ])

    # Initialize with 50x50 rectangular tile dimensions
    grid = PackedSpatialGrid(raw_points, tile_size=(50, 50))

    print(f"Grid Dimensions: {grid.num_tiles_x}x{grid.num_tiles_y}")
    print(f"LUT Shape: {grid.lut.shape}")
    print(f"Packed Buffer: \n{grid.point_buffer}")
    print(f"LUT (Start_Offset, Count): \n{grid.lut}")
    print(f"Maximum tile count: {grid.max_tile_count}")
    print("-" * 30)

    print(f"Tile index at (60, 10): {grid.getTileIndex(60, 10)}")
    print(f"Tile index at (10, 60): {grid.getTileIndex(10, 60)}")
    print(f"Tile index at (80, 80): {grid.getTileIndex(80, 80)}")
    print(f"Tile index out of bounds: {grid.getTileIndex(200, 200)}")

    for y_idx, x_idx in np.ndindex(grid.num_tiles_y, grid.num_tiles_x):
        tileIdx = np.array([x_idx, y_idx])
        print(f"Tile {tileIdx} ({grid.getTileCount(tileIdx)} points):\n{grid.getTilePoints(tileIdx)}")

    print("-" * 30)
    print("Box contained within one tile:")
    print(grid.getTileIndicesOverlappingBox(20, 20, 30, 30))

    print("Box overlapping four tiles:")
    print(grid.getTileIndicesOverlappingBox(40, 40, 70, 70))

    print("Box surrounding the entire grid:")
    print(grid.getTileIndicesOverlappingBox(-100, -100, 200, 200))

    print("Box outside the grid:")
    print(grid.getTileIndicesOverlappingBox(-100, -100, 0, 0))

    print("Box covering the extra empty column:")
    print(grid.getTileIndicesOverlappingBox(110, 10, 160, 160))

    # Verify sort_order: original indices [0,1,2,3,4,5] should map to packed order
    # Tile (0,0): point 0 -> packed[0]
    # Tile (1,0): points 1,3 -> packed[1], packed[2]
    # Tile (0,1): points 2,5 -> packed[3], packed[4]
    # Tile (1,1): point 4 -> packed[5]
    so = grid.sort_order
    print(f"\nSort order (original->packed indices): {so}")
    print(f"Original points:\n{raw_points}")
    print(f"Packed buffer (raw_points[sort_order]):\n{raw_points[so]}")
