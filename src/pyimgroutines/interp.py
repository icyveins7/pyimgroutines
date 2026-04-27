import numpy as np

class GridBilerp:
    """
    A custom bilinear interpolator class, to replace scipy's RegularGridInterpolator.
    The main issue with RegularGridInterpolator is that it is memory-intensive;
    it instantiates a meshgrid for the entire original image, which is by default a float64.
    This constitutes 2 new MxN float64 arrays (one for x and one for y), which is extremely wasteful.
    """
    def __init__(
        self,
        img: np.ndarray,
        x0: float = 0.0,
        xstep: float = 1.0,
        y0: float = 0.0,
        ystep: float = 1.0
    ):
        self._img = img
        self._x0 = x0
        self._xstep = xstep
        self._y0 = y0
        self._ystep = ystep

    def exec(self, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
        if xg.shape != yg.shape:
            raise ValueError("x and y must have the same shape")

        out = np.zeros(xg.shape, np.float64)
        it = np.nditer([xg, yg], flags=['multi_index'])
        for x, y in it:
            i = it.multi_index

            out[i] = self._bilerp(x, y) # pyright: ignore

        return out

    def _bilerp(self, x: float, y: float):
        xIdx = int(np.floor((x - self._x0) / self._xstep))
        yIdx = int(np.floor((y - self._y0) / self._ystep))

        # For now, use 0 for external values
        btmLeft = self._getValueWithDefault(yIdx, xIdx)
        btmRight = self._getValueWithDefault(yIdx, xIdx + 1)
        topLeft = self._getValueWithDefault(yIdx + 1, xIdx)
        topRight = self._getValueWithDefault(yIdx + 1, xIdx + 1)

        # Calculate fractional pixels
        xfrac = (x - self._x0) / self._xstep - xIdx
        yfrac = (y - self._y0) / self._ystep - yIdx

        # Interpolate top
        top = topLeft + (topRight - topLeft) * xfrac
        # Interpolate bottom
        bottom = btmLeft + (btmRight - btmLeft) * xfrac

        # y increases upwards
        return bottom + (top - bottom) * yfrac

    def _getValueWithDefault(self, yIdx: int, xIdx: int):
        if yIdx >= 0 and yIdx < self._img.shape[0] and xIdx >= 0 and xIdx < self._img.shape[1]:
            return self._img[yIdx, xIdx]
        else:
            return 0


if __name__ == "__main__":
    img = np.arange(2*2).reshape((2,2))
    print(img)
    interpolator = GridBilerp(img)
    print(interpolator.exec(np.array([0.5]), np.array([0.5])))

    x = np.arange(5) * 0.25 + 0
    y = np.arange(5) * 0.25 + 0
    xg, yg = np.meshgrid(x, y)
    print(xg)
    print(yg)
    print(interpolator.exec(xg, yg))

