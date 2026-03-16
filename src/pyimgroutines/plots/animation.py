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
        # NOTE: fps is an estimate, since the timer uses integer milliseconds
        self._animationFrame = 0 # reset
        self._animationTimer.timeout.connect(
            lambda: self._updateAnimations(fps)
        )
        self._animationTimer.start(int(1000.0 / fps)) # milliseconds

if __name__ == "__main__":
    fig = PgAnimation()
    fig.setPlotGrid(1, 2)
    import scipy as sp
    from .core import forceShow

    class CircleAnimation(ScatterAnimation):
        def at(self, frame: int, fps: int = 60):
            return np.array([np.cos(frame / (fps/2))]), np.array([np.sin(frame / (fps/2))])

    class SpiralAnimation(ScatterAnimation):
        def at(self, frame: int, fps: int = 60):
            frame = frame % 1000
            speed = 0.5
            r = frame / 1000
            theta = frame / (2 * np.pi) * speed
            return np.array([r * np.cos(theta)]), np.array([r * np.sin(theta)])

    class PolarCoordsAnimation(ScatterAnimation):
        def at(self, frame: int, fps: int = 60):
            frame = frame % 1000
            speed = 0.5
            r = frame / 1000
            theta = frame / (2 * np.pi) * speed
            theta = np.remainder(theta, 2*np.pi)
            return np.array([theta]), np.array([r])


    fig[0,0].addItem(CircleAnimation([0], [1]))
    fig[0,0].addItem(SpiralAnimation([0], [0]))
    fig[0,0].setXRange(-1.1, 1.1)
    fig[0,0].setYRange(-1.1, 1.1)
    fig[0,0].disableAutoRange(axis=pg.ViewBox.XYAxes)
    fig[0,0].setAspectLocked()
    fig[0,0].circles(np.array([0,0,1]), pg.mkPen("r"))
    # Overlay a simple image?
    img = np.arange(512*512).reshape(512, 512)
    fig[0,0].image(img, [-1,-1,2,2])

    fig[0,1].addItem(PolarCoordsAnimation())
    fig[0,1].disableAutoRange(axis=pg.ViewBox.XYAxes)
    fig[0,1].setYRange(-0.1, 1.1)
    fig[0,1].setXRange(-0.1, 2*np.pi+0.1)
    # Show equivalent remapped image
    x = np.linspace(-1, 1, 512)
    y = np.linspace(-1, 1, 512)
    interpolator = sp.interpolate.RegularGridInterpolator((x, y), img)
    r, theta = np.meshgrid(np.linspace(0,1,512), np.linspace(0,2*np.pi,512))
    xi = r * np.cos(theta)
    yi = r * np.sin(theta)
    pts = np.array([xi.flatten(), yi.flatten()]).T
    rimg = interpolator(pts)
    rimg = rimg.reshape(512, 512).T
    fig[0,1].image(rimg, [0,0,2*np.pi,1])

    fig.show()
    fig.animate(fps=60)

    forceShow()


