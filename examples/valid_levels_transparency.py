"""Show a checkerboard with alternating valid and transparent int32 pixels."""

import numpy as np
import pyqtgraph as pg

from pyimgroutines.plots import PgFigure, forceShow


rows, cols = np.indices((12, 12))
sentinel = np.iinfo(np.int32).max
data = np.zeros((12, 12), dtype=np.int32)
black = (rows + cols) % 2 == 1
data[black & (rows % 2 == 0)] = sentinel - 1
data[black & (rows % 2 == 1)] = sentinel
data.flags.writeable = False

fig = PgFigure(title="Integer sentinel transparency")
fig.plt.image(
    data,
    validLevels=(0, sentinel - 1),
    addHalfPixelBorder=False,
)
fig.plt.setTitle("White: 0 | Black: max-1")
fig.plt.setAspectLocked()

print(f"Image dtype: {fig.plt.im.image.dtype}")
print(f"Shares original data: {np.shares_memory(fig.plt.im.image, data)}")
print(f"Automatically chosen levels: {tuple(fig.plt.im.levels)}")

fig.show()
forceShow()
