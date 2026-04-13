from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core import PgPlotItem

import pyqtgraph as pg
import numpy as np

class HistogramItem(pg.BarGraphItem):
    def __init__(self, counts: np.ndarray | None = None, edges: np.ndarray | None = None, **kwargs):
        """
        A histogram item that can be added to a plot.
        Meant to work easily with np.histogram;
        see classmethod .fromData() to easily generate from data directly.

        NOTE: pen/brush is built-in to BarGraphItem, so pass those as per usual.
        Modifiations can be done via .setOpts(), which is also from BarGraphItem.

        Parameters
        ----------
        counts : np.ndarray
            The counts of the histogram.

        edges : np.ndarray
            The edges of the histogram.
        """
        self._counts = counts if counts is not None else np.array([0])
        self._edges = edges if edges is not None else np.array([0, 0])  
        width = self.calcWidth(self._edges)
        super().__init__(
            x0=self._edges[:-1],
            x1=self._edges[1:],
            height=self._counts,
            width=width,
            **kwargs
        )
        # self.setOpts(name="Histogram")

    @staticmethod
    def calcWidth(edges: np.ndarray):
        # Just use first 2 edges. Assume equally spaced.
        return (edges[1] - edges[0])

    @classmethod
    def fromData(cls, *args, **kwargs):
        """
        Redirects to np.histogram. See np.histogram for more information.
        """
        counts, edges = np.histogram(*args, **kwargs)
        return cls(counts, edges)

    def updateBins(self, data: np.ndarray):
        # TODO: allow outside setting of bins
        counts, edges = np.histogram(data)
        self.setOpts(x0=edges[:-1], x1=edges[1:], height=counts, width=self.calcWidth(edges))

    def linkToImagePlot(self, plt: PgPlotItem):
        plt.sigROIselectionChangeFinished.connect(self.updateBins)


if __name__ == "__main__":
    from ..core import PgFigure, forceShow
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
