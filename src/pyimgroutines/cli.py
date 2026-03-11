import argparse
import numpy as np
from .plots import PgFigure, forceShow

def PlotImageFile():
    parser = argparse.ArgumentParser(
        description="Plot a binary image file using PgFigure. Currently only supports single-channel (grayscale) images."
    )
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Path to the binary image file")
    parser.add_argument("-W", "--width", type=int, required=True, help="Width of the image in pixels")
    parser.add_argument("-H", "--height", type=int, required=True, help="Height of the image in pixels")
    parser.add_argument(
        "--dtype", type=np.dtype, default=np.float32,
        help="NumPy dtype of the data (default: float32)"
    )
    parser.add_argument(
        "--offsetBytes", type=int, default=0,
        help="Byte offset to the first image byte (default: 0)"
    )
    parser.add_argument(
        "--aspectLocked", action="store_true",
        help="Lock aspect ratio (default: False)"
    )
    args = parser.parse_args()

    # Load raw binary data with offset
    data = np.fromfile(args.filepath, dtype=args.dtype, offset=args.offsetBytes)
    data = data.reshape((args.height, args.width))

    # Plot
    fig = PgFigure()
    fig.plt.image(data)
    if args.aspectLocked:
        fig.plt.setAspectLocked()
    fig.show()
    forceShow() # otherwise the window will appear and then disappear immediately
