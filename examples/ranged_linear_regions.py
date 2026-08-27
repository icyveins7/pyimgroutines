"""
Example demonstrating the rangedLinearRegion() method.

This example creates a simple sine wave and adds immovable linear regions
where the y-values exceed a specified threshold.
"""

import numpy as np
import pyqtgraph as pg
from pyimgroutines.plots import PgFigure, forceShow

# Create sample data - a sine wave with some noise
x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

# Create figure and plot the data
fig = PgFigure(title="Ranged Linear Regions Example, use hotkeys directly")
fig.plt.plot(x, y, pen=pg.mkPen("b", width=2))


fig.show()
forceShow()
