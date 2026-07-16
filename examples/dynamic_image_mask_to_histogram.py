"""
This example shows 2 interactive methods:

1) How to use a rectangular ROI over an image to create a histogram.
2) How to use a linear region (over a histogram, for visual reference) to create a mask of an image.

The original image gets plotted in subplot 1.
This is where you can toggle the ROI (with letter 'R' hotkey), which is connected to a
dynamically updated (grey) histogram in subplot 2.

The histogram (generated once, just for visual reference) is in subplot 2.
If the ROI is toggled on in subplot 1, you should also see a 2nd, grey histogram in subplot 2;
this histogram is dynamically generated from the pixels captured in the ROI.

The mask (based on the linear region item in subplot 2) is dynamically updated in subplot 3.
"""

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
