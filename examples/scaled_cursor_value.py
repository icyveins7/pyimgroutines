"""Press V once to show scaled int16 pixel values without changing the image."""

import numpy as np

from pyimgroutines.plots import PgFigure, forceShow


data = np.arange(12 * 12, dtype=np.int16).reshape(12, 12)
scale = np.float64(1.234)

fig = PgFigure(title="Scaled int16 cursor values")
fig.plt.image(data, addHalfPixelBorder=False)
fig.plt.setCursorValueFormatter(
    lambda row, col, value: (
        f"[{row}, {col}]: {np.float64(value) * scale:.3f} (raw: {value})"
    )
)
fig.plt.setAspectLocked()
fig.plt.setTitle("Press V once, then hover over the image")
fig.show()
forceShow()
