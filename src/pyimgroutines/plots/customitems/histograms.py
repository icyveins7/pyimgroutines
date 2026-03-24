import pyqtgraph as pg
import numpy as np

class HistogramItem(pg.BarGraphItem):
    def __init__(self, counts: np.ndarray, edges: np.ndarray, **kwargs):
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
        self._counts = counts
        self._edges = edges
        width = self.calcWidth(self._edges)
        super().__init__(
            x0=edges[:-1],
            x1=edges[1:],
            height=counts,
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



if __name__ == "__main__":
    from ..core import PgFigure, forceShow
    fig = PgFigure()
    fig.setPlotGrid(2,1)
    x = np.random.randint(0, 10, 100).reshape((10, 10))
    fig[0,0].image(x)
    hist = HistogramItem.fromData(x)
    fig[1,0].addItem(hist)
    hist.setOpts(pen=pg.mkPen("k"), brush=pg.mkBrush("r"))
    fig.show()
    forceShow()
