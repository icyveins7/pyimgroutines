import numpy as np

from pyimgroutines.plots import PgFigure, forceShow
from pyimgroutines.point_raster import points_to_image


points = np.random.randn(10000, 2)
image, xywh = points_to_image(points)

fig = PgFigure()
fig.setPlotGrid(2, 1, linkX=True, linkY=True, aspectLocked=True)

fig[0, 0].image(
    image,
    xywh=xywh,
    addHalfPixelBorder=False,
)
fig[1, 0].scatterPlot(
    points[:, 0],
    points[:, 1],
    pen=None,
    symbol="o",
    symbolBrush="r",
)

fig.show()
forceShow()
