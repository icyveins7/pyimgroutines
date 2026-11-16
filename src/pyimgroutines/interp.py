import numpy as np

class GridBilerp:
    """
    A custom bilinear interpolator class, to replace scipy's RegularGridInterpolator.
    The main issue with RegularGridInterpolator is that it is memory-intensive;
    it instantiates a meshgrid for the entire original image, which is by default a float64.
    This constitutes 2 new MxN float64 arrays (one for x and one for y), which is extremely wasteful.

    Set nanOutside=True to return NaN for queries outside the image instead of
    using zero-valued external samples.
    """
    def __init__(
        self,
        img: np.ndarray,
        x0: float = 0.0,
        xstep: float = 1.0,
        y0: float = 0.0,
        ystep: float = 1.0,
        nanOutside: bool = False,
    ):
        self._img = img
        self._x0 = x0
        self._xstep = xstep
        self._y0 = y0
        self._ystep = ystep
        self._nanOutside = nanOutside

    def exec(self, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
        if xg.shape != yg.shape:
            raise ValueError("x and y must have the same shape")

        return self._bilerpNumpy(xg, yg)

    def _bilerpNumpy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly sample an image at arrays of query coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates of the output grid.
        y : np.ndarray
            Y coordinates of the output grid, with the same shape as ``x``.

        Returns
        -------
        np.ndarray
            Floating-point interpolated values with the grid's shape.
        """
        xPos = (x - self._x0) / self._xstep
        yPos = (y - self._y0) / self._ystep
        if self._nanOutside:
            valid = (xPos >= 0) & (xPos <= self._img.shape[1] - 1) & (yPos >= 0) & (yPos <= self._img.shape[0] - 1)
            xPos = xPos[valid]
            yPos = yPos[valid]
        xIdx = np.floor(xPos).astype(np.intp)
        yIdx = np.floor(yPos).astype(np.intp)
        xfrac = xPos - xIdx
        yfrac = yPos - yIdx

        def values(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
            """Read neighbouring values as floats, using zero outside the image."""
            inside = (rows >= 0) & (rows < self._img.shape[0]) & (cols >= 0) & (cols < self._img.shape[1])
            result = self._img[
                np.clip(rows, 0, self._img.shape[0] - 1),
                np.clip(cols, 0, self._img.shape[1] - 1),
            ].astype(np.float64)
            result[~inside] = 0
            return result

        bottomLeft = values(yIdx, xIdx)
        bottomRight = values(yIdx, xIdx + 1)
        topLeft = values(yIdx + 1, xIdx)
        topRight = values(yIdx + 1, xIdx + 1)
        lower = bottomLeft + (bottomRight - bottomLeft) * xfrac
        upper = topLeft + (topRight - topLeft) * xfrac
        result = lower + (upper - lower) * yfrac
        if self._nanOutside:
            image = np.full(x.shape, np.nan)
            image[valid] = result
            return image
        return result

    def _bilerp(self, x: float, y: float):
        xPos = (x - self._x0) / self._xstep
        yPos = (y - self._y0) / self._ystep
        if self._nanOutside and (xPos < 0 or xPos > self._img.shape[1] - 1 or yPos < 0 or yPos > self._img.shape[0] - 1):
            return np.nan
        xIdx = int(np.floor(xPos))
        yIdx = int(np.floor(yPos))

        # For now, use 0 for external values
        btmLeft = float(self._getValueWithDefault(yIdx, xIdx))
        btmRight = float(self._getValueWithDefault(yIdx, xIdx + 1))
        topLeft = float(self._getValueWithDefault(yIdx + 1, xIdx))
        topRight = float(self._getValueWithDefault(yIdx + 1, xIdx + 1))

        # Calculate fractional pixels
        xfrac = xPos - xIdx
        yfrac = yPos - yIdx

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

