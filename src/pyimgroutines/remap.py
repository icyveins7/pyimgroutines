from scipy.interpolate import RegularGridInterpolator
import numpy as np

def interpWhileKeepingEdge(
    src: np.ndarray,
    numRows: int,
    numCols: int,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to perform interpolation via
    RegularGridInterpolator while maintaining the pixel border locations
    i.e. the start/end x and y values are the same after this.

    Parameters
    ----------
    src : np.ndarray
        Input image.

    numRows : int
        Number of rows in destination image.

    maxCols : int
        Number of columns in destination image.

    Returns
    -------
    result : np.ndarray
        Destination image.

    dstY : np.ndarray
        Destination image Y mesh values.

    dstX : np.ndarray
        Destination image X mesh values.
    """
    srcX = np.arange(src.shape[1])
    srcY = np.arange(src.shape[0])
    dstXstep = (src.shape[0] - 1) / (numCols - 1)
    dstYstep = (src.shape[1] - 1) / (numRows - 1)
    dstY, dstX = np.meshgrid(
        np.arange(numRows) * dstYstep,
        np.arange(numCols) * dstXstep
    )

    interp = RegularGridInterpolator((srcX, srcY), src, **kwargs)
    dst = interp((dstX, dstY))
    return dst, dstY, dstX

if __name__ == "__main__":
    from .plots import PgFigure, forceShow

    fig = PgFigure()
    fig.setPlotGrid(1,2)
    img = np.arange(64 * 64).reshape((64, 64))
    fig[0,0].image(img)
    rmp, dstY, dstX = interpWhileKeepingEdge(img, 8, 8)
    print(dstY)
    print(dstX)
    fig[0,1].image(rmp, xMesh_yMesh=(dstX,dstY))
    fig.show()
    forceShow()
