from .core import PgFigure

import numpy as np

from PySide6.QtCore import QTimer

import pyqtgraph as pg

class ScatterAnimation(pg.ScatterPlotItem):
    def at(self, frame: int, fps: int = 60):
        raise NotImplementedError("Should be implemented by subclass")

class PgAnimation(PgFigure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._animationFrame = 0
        self._animationTimer = QTimer()

    def _updateAnimations(self, fps: int):
        for plt in np.nditer(self.plts, ['refs_ok']):
            for dataItem in plt.item().listDataItems(): # pyright: ignore
                if isinstance(dataItem, ScatterAnimation):
                    x, y = dataItem.at(self._animationFrame, fps=fps)
                    # print(x, y)
                    dataItem.setData(x=x, y=y)

        self._animationFrame += 1
        # print(self._animationFrame)

    def animate(self, fps: int = 60):
        self._animationFrame = 0 # reset
        self._animationTimer.timeout.connect(
            lambda: self._updateAnimations(fps)
        )
        self._animationTimer.start(int(1000.0 / fps)) # milliseconds

if __name__ == "__main__":
    fig = PgAnimation()
    import numpy as np
    from .core import forceShow

    class CircleAnimation(ScatterAnimation):
        def at(self, frame: int, fps: int = 60):
            return np.array([np.cos(frame / (fps/2))]), np.array([np.sin(frame / (fps/2))])

    class SpiralAnimation(ScatterAnimation):
        def at(self, frame: int, fps: int = 60):
            return np.array([np.cos(frame / (fps/2))]), np.array([np.sin(frame / (fps/2))])


    fig.plt.addItem(CircleAnimation([0], [1]))
    fig.plt.setXRange(-1.1, 1.1)
    fig.plt.setYRange(-1.1, 1.1)
    fig.plt.disableAutoRange(axis=pg.ViewBox.XYAxes)
    fig.plt.setAspectLocked()
    fig.plt.circles(np.array([0,0,1]), pg.mkPen("r"))
    fig.show()
    fig.animate(fps=60)

    forceShow()


