from pyimgroutines.plots.core import PgFigure, forceShow
from pyimgroutines.plots.customitems import HistogramItem

import numpy as np
import pyqtgraph as pg

fig = PgFigure()
fig.setPlotGrid(3,1)
x = np.random.randint(0, 10, 100).reshape((10, 10))
fig[0,0].image(x)
hist = HistogramItem.fromData(x)
fig[1,0].addItem(hist)
hist.setOpts(pen=pg.mkPen("k"), brush=pg.mkBrush("r"))

dynhist = HistogramItem()
dynhist.linkToImagePlot(fig[0,0])
fig[1,0].addItem(dynhist)

regionItem = pg.LinearRegionItem()
fig[1,0].addItem(regionItem)

fig[2,0].image(fig[0,0].mask)
fig[0,0].linkToLinearRegionItem(regionItem)
def updateMaskImg(mask):
    fig[2,0].im.setImage(mask.astype(np.uint8))
fig[0,0].sigMaskChanged.connect(updateMaskImg)

fig.show()
forceShow()
