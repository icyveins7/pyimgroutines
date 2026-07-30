"""
Experiments to get fast/responsive hover signal/slots on high density scatter plots.
"""

from pyimgroutines.plots import PgFigure, closeAllFigs, forceShow
from pyimgroutines.plots.customitems import NearestScatterPlotItem
closeAllFigs()
import numpy as np
import pyqtgraph as pg

# At this number of points, rendering overhead still dominates
# everything.
length = 500000
x = np.random.randn(length)
y = np.random.randn(length)

hoverOverlays = {}

def onSigHovered(item, pts, evt):
    key = id(item)
    if key not in hoverOverlays:
        overlay = pg.ScatterPlotItem(
            symbol='o',
            size=10,
            brush='w',
            pen=None,
        )
        overlay.setAcceptHoverEvents(False)
        overlay.setZValue(100)
        item.getViewBox().addItem(overlay)
        hoverOverlays[key] = overlay

    overlay = hoverOverlays[key]
    positions = np.array([
        [point.pos().x(), point.pos().y()]
        for point in pts
    ])
    overlay.setData(
        x=positions[:, 0] if len(positions) else [],
        y=positions[:, 1] if len(positions) else [],
    )

f = PgFigure()
item = pg.ScatterPlotItem(
    x=x,
    y=y,
    brush='r',
    size=10,
    hoverable=True,
    tip=None,
)
item.sigHovered.connect(onSigHovered)
f.plt.addItem(item)

f.show()

f2 = PgFigure()
item2 = NearestScatterPlotItem(
    x=x,
    y=y,
    brush='r',
    size=10,
    hoverable=True,
    tip=None,
)
item2.sigHovered.connect(onSigHovered)
f2.plt.addItem(item2)
f2.show()

forceShow()
