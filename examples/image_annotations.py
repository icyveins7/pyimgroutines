import numpy as np

from pyimgroutines.plots import PgFigure, forceShow


image = np.arange(100, dtype=np.float32).reshape(10, 10)

fig = PgFigure(title="Image annotations")
fig.plt.image(image, cmap=None)
fig.plt.setAspectLocked()

# Each annotation snaps to the centre of the nearest image pixel. The box
# remains the same size on screen when the image is zoomed.
fig.plt.annotate((2, 3))
fig.plt.annotate((5, 7))

fig.show()
forceShow()
