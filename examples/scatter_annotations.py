"""
Annotate scatter points with the 'x' hotkey.

Hover a point and press 'x' to toggle an annotation box for it; press 'gd'
to clear all annotations. The label text is fully user-defined via
setAnnotationFormatter(), and is only evaluated when the point is annotated,
so expensive lookups can be deferred until they are needed.
"""

import numpy as np

from pyimgroutines.plots import PgFigure, forceShow

rng = np.random.default_rng(0)
n = 200
x = rng.normal(size=n)
y = rng.normal(size=n)
# Some per-point values that are NOT stored on the scatter item; the
# formatter looks them up on demand (this could equally be a database call)
values = rng.uniform(0, 100, size=n)

fig = PgFigure(title="Scatter annotations")
fig.plt.setAspectLocked()

# Default formatter: shows X,Y and the array index of the point
item = fig.plt.scatter(x, y, brush="r", pen=None, size=10, name="default")

# Custom formatter: index is the position in the arrays passed to scatter()
item2 = fig.plt.scatter(x + 4, y, brush="c", pen=None, size=10, name="custom")
item2.setAnnotationFormatter(
    lambda index, px, py, data: f"#{index}\nX: {px:.3f}, Y: {py:.3f}\nValue: {values[index]:.1f}"
)

# Per-point payload via data=; available as the formatter's 4th argument
labels = [f"pt-{i:03d}" for i in range(n)]
item3 = fig.plt.scatter(x + 8, y, brush="y", pen=None, size=10, data=labels, name="payload")
item3.setAnnotationFormatter(lambda index, px, py, data: f"{data}\n({px:.2f}, {py:.2f})")

# Annotations can also be added programmatically
fig.plt.annotateScatterPoint(item2, 0)

fig.plt.addLegend()
fig.show()
forceShow()
