import argparse
import numpy as np
from .plots import PgFigure, forceShow
from .io import read_image_with_header

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
    print(f"Reading {args.filepath}...")
    data, header = read_image_with_header(args.filepath, args.width, args.height, args.dtype, args.offsetBytes)
    print("Done")
    print("Header bytes: ")
    print(header)

    # Estimate colorbar levels using percentiles (robust to invalid values and NaNs)
    lower, upper = np.nanpercentile(data, [35, 65])

    # Plot
    fig = PgFigure(title=f"{args.filepath}")
    fig.plt.image(data)
    fig.plt.cbar.setLevels((lower, upper))
    if args.aspectLocked:
        fig.plt.setAspectLocked()
    fig.show()
    forceShow() # otherwise the window will appear and then disappear immediately

def CompareImageFiles():
    parser = argparse.ArgumentParser(
        description="Compare two binary image files side-by-side with their difference. Currently only supports single-channel (grayscale) images."
    )
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Comma-separated paths to two binary image files (e.g. file1.bin,file2.bin)")
    parser.add_argument("-W", "--width", type=int, required=True, help="Width of the images in pixels")
    parser.add_argument("-H", "--height", type=int, required=True, help="Height of the images in pixels")
    parser.add_argument(
        "--dtype", type=str, default="float32,float32",
        help="Comma-separated NumPy dtypes for each image (default: float32,float32)"
    )
    parser.add_argument(
        "--offsetBytes", type=str, default="0,0",
        help="Comma-separated byte offsets to the first image byte for each file (default: 0,0)"
    )
    parser.add_argument(
        "--aspectLocked", action="store_true",
        help="Lock aspect ratio (default: False)"
    )
    args = parser.parse_args()

    # Parse comma-separated arguments
    filepaths = args.filepath.split(',')
    if len(filepaths) != 2:
        raise ValueError("Exactly two comma-separated filepaths are required")

    dtypes = args.dtype.split(',')
    if len(dtypes) != 2:
        raise ValueError("Exactly two comma-separated dtypes are required")
    dtypes = [np.dtype(d) for d in dtypes]

    offsets = args.offsetBytes.split(',')
    if len(offsets) != 2:
        raise ValueError("Exactly two comma-separated offsetBytes are required")
    offsets = [int(o) for o in offsets]

    # Load raw binary data with offset
    print(f"Reading {filepaths[0]}...")
    data1, header1 = read_image_with_header(filepaths[0], args.width, args.height, dtypes[0], offsets[0])
    print("Done.")
    print(f"Header bytes from {filepaths[0]}: ")
    print(header1)

    print(f"Reading {filepaths[1]}...")
    data2, header2 = read_image_with_header(filepaths[1], args.width, args.height, dtypes[1], offsets[1])
    print("Done.")
    print(f"Header bytes from {filepaths[1]}: ")
    print(header2)

    # Compute difference (promote to appropriate float type)
    result_dtype = np.result_type(data1, data2, np.float32)
    diff = data1.astype(result_dtype) - data2.astype(result_dtype)

    # Estimate colorbar levels using percentiles (robust to invalid values and NaNs)
    lower_1, upper_1 = np.nanpercentile(data1, [35, 65])
    lower_2, upper_2 = np.nanpercentile(data2, [35, 65])
    lower_d, upper_d = np.nanpercentile(diff,  [35, 65])
    if lower_d == upper_d:
        lower_d = -1
        upper_d = 1

    # Plot 1x3 grid
    fig = PgFigure(title=f"{filepaths[0]} vs {filepaths[1]}")
    fig.setPlotGrid(1, 3, linkX=True, linkY=True, aspectLocked=args.aspectLocked)
    fig[0, 0].image(data1)
    fig[0, 0].cbar.setLevels((lower_1, upper_1))
    fig[0, 1].image(data2)
    fig[0, 1].cbar.setLevels((lower_2, upper_2))
    fig[0, 2].image(diff)
    fig[0, 2].cbar.setLevels((lower_d, upper_d))
    fig.show()
    forceShow()
