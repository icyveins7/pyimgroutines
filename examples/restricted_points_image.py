import numpy as np

from pyimgroutines.plots import PgFigure, forceShow
from pyimgroutines.point_raster import points_to_image

# --- Example 1: Dense cloud of points with restricted scatter ---

points = np.random.randn(500000, 2)

fig = PgFigure()
restricted = fig.plt.restrictedscatter(
    points,
    tile_size=(0.2, 0.2),
    max_tile_span=3,
    symbol="o",
    brush="r",
)

def on_tiles_changed():
    idx = restricted.active_tile_indices
    print(idx)

restricted.sigTilesChanged.connect(on_tiles_changed)

fig.plt.rectangle([np.min(points[:, 0]), np.min(points[:, 1])],
                  [np.max(points[:,0])-np.min(points[:, 0]), np.max(points[:, 1])-np.min(points[:, 1])])
print(f"Restricted scatter: {restricted.grid.num_tiles_y} x {restricted.grid.num_tiles_x} tiles")
print(f"Max tile count: {restricted.grid.max_tile_count}")
fig.show()

forceShow()
