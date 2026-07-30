from pyimgroutines.plots import PgFigure, closeAllFigs, forceShow
from pyimgroutines.plots.customitems import NearestScatterPlotItem
closeAllFigs()
import numpy as np

x = np.random.randn(100000)
y = np.random.randn(100000)

def onSigHovered(item, pts, evt):
    if len(pts) != 0:
        print(pts)
        print(evt)

        # pt is a SpotItem
        pt = pts[0]
        print(pt.index())
        print(f"{x[pt.index()]}, {y[pt.index()]}")

f = PgFigure()
item = f.plt.scatterPlot(x, y, brush='r')
item.scatter.setData(hoverable=True, hoverBrush="w")
item.sigPointsHovered.connect(onSigHovered)

f.show()

f2 = PgFigure()
item2 = NearestScatterPlotItem(
    x=x,
    y=y,
    brush='r',
    size=10,
    hoverable=True,
    hoverBrush='w',
)
item2.sigHovered.connect(onSigHovered)
f2.plt.addItem(item2)
f2.show()

forceShow()
