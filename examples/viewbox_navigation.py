import numpy as np
import pyqtgraph as pg

from pyimgroutines.aabb import bounding_box
from pyimgroutines.plots import PgFigure, forceShow


rng = np.random.default_rng()
groups = [rng.random((rng.integers(2, 6), 2)) for _ in range(8)]

fig = PgFigure(title="Predefined view boxes")
fig.plt.addLegend()
for group_index, points in enumerate(groups):
    fig.plt.scatterPlot(
        points[:, 0],
        points[:, 1],
        pen=None,
        symbol="o",
        brush=pg.intColor(group_index),
        name=f"Group {group_index + 1}",
    )


def view_box(index):
    return bounding_box(groups[index], padding_factor=0.25), len(groups)


fig.plt.setViewBoxFunction(view_box)
fig.show()
forceShow()
