import numpy as np

from pyimgroutines.plots import PgFigure, forceShow
from pyimgroutines.point_raster import points_to_image


points = np.random.randn(500000, 2)
image, xywh = points_to_image(points)

fig = PgFigure()
# fig.setPlotGrid(2, 1, linkX=True, linkY=True, aspectLocked=True)

# fig[0, 0].image(
#     image,
#     xywh=xywh,
#     addHalfPixelBorder=False,
#     cmap = None,
#     colorbar = False,
# )
fig.plt.scatterPlot(
    points[:, 0],
    points[:, 1],
    pen=None,
    symbol="o",
    symbolBrush="r",
)

fig.show()

fig2 = PgFigure()
# fig2.setPlotGrid(2,1,True,True,True)
hybrid = fig2[0,0].hybridscatter(
    points,
    tile_size=(0.25, 0.25),
    img_dims=(4096, 4096),
)
print(f"Using {hybrid.grid.num_tiles_y} x {hybrid.grid.num_tiles_x} tiles")
print(f"Max tile count: {hybrid.grid.max_tile_count}")
# fig2[1,0].scatterPlot(
#     points[:, 0],
#     points[:, 1],
#     pen=None,
#     symbol="o",
#     symbolBrush="r",
# )
fig2.show()

forceShow()
