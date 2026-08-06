import numpy as np
import pyqtgraph as pg

from pyimgroutines.aabb import bounding_box
from pyimgroutines.plots import PgFigure, forceShow


rng = np.random.default_rng()
groups = [rng.random((rng.integers(2, 6), 2)) for _ in range(8)]
groups_by_plot = (groups[:4], groups[4:])

fig = PgFigure(title="Predefined view boxes")
fig.setPlotGrid(1, 2)

for plot_index, plot_groups in enumerate(groups_by_plot):
    plt = fig[0, plot_index]
    plt.addLegend()
    for group_index, points in enumerate(plot_groups):
        global_group_index = plot_index * 4 + group_index
        plt.scatterPlot(
            points[:, 0],
            points[:, 1],
            pen=None,
            symbol="o",
            brush=pg.intColor(global_group_index),
            name=f"Group {global_group_index + 1}",
        )

    def view_box(index, groups=plot_groups):
        return bounding_box(groups[index], padding_factor=0.25), len(groups)

    plt.setViewBoxFunction(view_box)

fig.show()
forceShow()
