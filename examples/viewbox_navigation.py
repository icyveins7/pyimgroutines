import numpy as np

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
        symbolBrush="r",
        name=f"Group {group_index + 1}",
    )


def view_box(index):
    points = groups[index]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, 0.05)
    margin = 0.25 * span
    view_box = ((lower[0] - margin[0], upper[0] + margin[0]),
                (lower[1] - margin[1], upper[1] + margin[1]))
    return view_box, len(groups)


fig.plt.setViewBoxFunction(view_box)
fig.show()
forceShow()
