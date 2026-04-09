import numpy as np

def read_image_with_header(
    filepath: str,
    width: int,
    height: int = 1,
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

    width : int
        Width of image in pixels.

    height : int
        Height of image in pixels.

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
    imgBytes = width * height * np.dtype(imgDtype).itemsize
    img = data[offsetBytes:offsetBytes + imgBytes].view(imgDtype)
    img = img.reshape((height, width))
    return img, header
