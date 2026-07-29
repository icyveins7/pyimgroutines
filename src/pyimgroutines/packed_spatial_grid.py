import numpy as np

class PackedSpatialGrid:
    def __init__(self, points, tile_size):
        """
        Initialize and 'bake' the spatial grid.
        :param points: A numpy array of shape (N, 2) representing (x, y) coordinates.
        :param tile_size: The width and height of each square tile.
        """
        self.points = points.astype(np.float64)
        self.tile_size = tile_size
        
        # Determine grid dimensions based on point bounds
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        
        # Number of tiles needed to cover the extent of the points
        num_tiles_x = int(np.ceil((max_coords[0] - min_coords[0]) / tile_size)) + 1
        num_tiles_y = int(np.ceil((max_coords[1] - min_coords[1]) / tile_size)) + 1
        
        self.num_tiles_x = num_tiles_x
        self.num_tiles_y = num_tiles_y
        self.total_tiles = num_tiles_x * num_tiles_y

        # 1. Calculate tile indices for every point
        # We normalize coordinates so the minimum point starts at (0,0)
        normalized_points = self.points - min_coords
        x_indices = (normalized_points[:, 0] // tile_size).astype(int)
        y_indices = (normalized_points[:, 1] // tile_size).astype(int)
        
        # Tile index = x_index + (y_index * width_of_grid)
        # This allows for a 1D LUT where index i corresponds to a specific (x,y)
        tile_indices = x_indices + (y_indices * num_tiles_x)
        
        # 2. Sort points by their tile index to create the packed buffer
        sorted_indices = np.argsort(tile_indices)
        self.point_buffer = self.points[sorted_indices]
        sorted_tile_ids = tile_indices[sorted_indices]
        
        # 3. Build the Look-Up Table (LUT)
        # LUT will store [start_offset, count] for each tile index
        self.lut = np.zeros((self.total_tiles, 2), dtype=np.int32)
        
        # Find unique tile indices and their counts in the sorted list
        unique_ids, counts = np.unique(sorted_tile_ids, return_counts=True)
        
        # Calculate cumulative start offsets
        offsets = np.zeros(len(unique_ids), dtype=np.int32)
        offsets[1:] = np.cumsum(counts)[:-1]
        
        # Populate the LUT
        # Note: Tiles with no points will have [0, 0] or the offset of the next tile?
        # To keep it O(1), we want the exact index. If a tile is empty, count is 0.
        for i, tid in enumerate(unique_ids):
            self.lut[tid] = [offsets[i], counts[i]]

    def query(self, x, y):
        """
        Retrieve points for a specific subgrid coordinate.
        """
        # Normalize input coordinate
        min_coords = np.min(self.points, axis=0)
        norm_x = x - min_coords[0]
        norm_y = y - min_coords[1]
        
        x_idx = int(norm_x // self.tile_size)
        y_idx = int(norm_y // self.tile_size)
        
        # Safety check for bounds
        if x_idx >= self.num_tiles_x or y_idx >= self.num_tiles_y or x_idx < 0 or y_idx < 0:
            return np.array([])

        tile_idx = x_idx + (y_idx * self.num_tiles_x)
        
        start_offset = self.lut[tile_idx, 0]
        count = self.lut[tile_idx, 1]
        
        if count == 0:
            return np.array([])
        
        return self.point_buffer[start_offset : start_offset + count]

# --- Smoke Test ---

if __name__ == "__main__":
    # Define points that follow the user's 2x2 logic:
    # Tile (0,0): 1 point
    # Tile (1,0): 2 points
    # Tile (0,1): 2 points
    # Tile (1,1): 1 point
    # Total: 6 points
    raw_points = np.array([
        [10, 10],   # Tile (0,0) - Index 0
        [60, 10],   # Tile (1,0) - Index 1
        [70, 10],   # Tile (1,0) - Index 1
        [10, 60],   # Tile (0,1) - Index 2
        [10, 70],   # Tile (0,1) - Index 2
        [80, 80]    # Tile (1,1) - Index 3
    ])

    # Initialize with 50x50 tile size
    grid = PackedSpatialGrid(raw_points, tile_size=50)

    print(f"Grid Dimensions: {grid.num_tiles_x}x{grid.num_tiles_y}")
    print(f"LUT Shape: {grid.lut.shape}")
    print(f"Packed Buffer: \n{grid.point_buffer}")
    print(f"LUT (Start_Offset, Count): \n{grid.lut}")
    print("-" * 30)

    # Test Querying Tile (1,0) -> Index 1
    # Expected: 2 points ([60, 10], [70, 10])
    res1 = grid.query(60, 10)
    print(f"Query Tile (1,0) at (60, 10): \n{res1}")

    # Test Querying Tile (0,1) -> Index 2
    # Expected: 2 points ([10, 60], [10, 70])
    res2 = grid.query(10, 60)
    print(f"Query Tile (0,1) at (10, 60): \n{res2}")

    # Test Querying Tile (1,1) -> Index 3
    # Expected: 1 point ([80, 80])
    res3 = grid.query(80, 80)
    print(f"Query Tile (1,1) at (80, 80): \n{res3}")

    # Test Querying Empty Tile (0,0) is actually populated in our data, 
    # but let's try a coordinate outside the data bounds to ensure safety.
    res4 = grid.query(200, 200)
    print(f"Query Out of Bounds: \n{res4}")
