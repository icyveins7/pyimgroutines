import numpy as np

def read_image_with_header(
    filepath: str,
    width: int | tuple[int, type | np.dtype],
    height: int | tuple[int, type | np.dtype] = 1,
    imgDtype: type | np.dtype = np.uint8,
    offsetBytes: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to read and reshape image data
    into an appropriate array from a file.
    Assumes single-channel image data.
    Also returns the header bytes.

    Parameters
    ----------
    filepath : str
        Image filepath to read.

    width : int | tuple[int, type]
        Width of image in pixels.
        You may use a tuple to specify this if it should be read from the header;
        e.g. (5, np.int32) will read bytes 5:9 (4 bytes) as an int32 as the width.

    height : int | tuple[int, type]
        Height of image in pixels.
        You may use a tuple to specify this if it should be read from the header;
        e.g. (10, np.int32) will read bytes 10:14 (4 bytes) as an int32 as the height.

    imgDtype : type
        Image pixel data type.

    offsetBytes : int
        Byte offset to the first image byte.
        All bytes from the beginning of the file
        to this are considered the header.

    Returns
    -------
    img : np.ndarray
        Image array with shape (height, width)
        and type imgDtype.

    header : np.ndarray
        Header bytes of type uint8.
    """
    data = np.fromfile(filepath, dtype=np.uint8)
    header = data[:offsetBytes]
    if isinstance(width, tuple):
        widthOffsetBytes, widthType = width
        width = header[widthOffsetBytes:widthOffsetBytes + np.dtype(widthType).itemsize].view(widthType)[0]
        print(f"Parsed width from header: {width}")
    if isinstance(height, tuple):
        heightOffsetBytes, heightType = height
        height = header[heightOffsetBytes:heightOffsetBytes + np.dtype(heightType).itemsize].view(heightType)[0]
        print(f"Parsed height from header: {height}")

    imgBytes = width * height * np.dtype(imgDtype).itemsize # pyright:ignore
    img = data[offsetBytes:offsetBytes + imgBytes].view(imgDtype) # pyright:ignore
    img = img.reshape((height, width)) # pyright:ignore
    return img, header
