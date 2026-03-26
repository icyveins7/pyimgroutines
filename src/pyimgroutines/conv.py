import scipy as sp
import scipy.ndimage as spi
import numpy as np

class RealConvolver:
    def __init__(self, img_dims: tuple, kernel: np.ndarray, img_dtype: type = np.float64, correlateInstead: bool = False):
        self._img_dims = img_dims
        self._kernel = kernel
        targetDims = (img_dims[0] + kernel.shape[0] - 1, img_dims[1] + kernel.shape[1] - 1)
        self._kernelfft = self.precomputeKernelTransform(self._kernel, targetDims, correlateInstead)
        self._imgpad = np.zeros(targetDims, dtype=img_dtype)

    def precomputeKernelTransform(self, kernel: np.ndarray, final_dims: tuple, correlateInstead: bool) -> np.ndarray:
        kernelpad = np.zeros((final_dims[0], final_dims[1]), dtype=kernel.dtype)
        kernelpad[0:kernel.shape[0], 0:kernel.shape[1]] = kernel
        kernelfft = sp.fft.rfft2(kernelpad)
        if correlateInstead:
            print("Correlating instead of convolving")
            kernelfft = np.conj(kernelfft)
        return kernelfft

    def exec(self, img: np.ndarray, clip: bool = True) -> np.ndarray:
        self._imgpad[:] = 0
        self._imgpad[0:img.shape[0], 0:img.shape[1]] = img
        # NOTE: must set shape for irfft2, since it cannot be inferred exactly
        result = sp.fft.irfft2(self._kernelfft * sp.fft.rfft2(self._imgpad), s=self._imgpad.shape)
        if clip:
            i0 = self._kernel.shape[0] // 2
            i1 = self._kernel.shape[1] // 2
            result = result[i0:-i0, i1:-i1]
        return result

if __name__ == "__main__":
    shape = (7, 7)
    x = np.random.randint(0,9,shape)
    print(x)
    kern = np.random.randint(0,2,(5,5))
    print(kern)

    print("---------")
    conv = RealConvolver(shape, kern)

    y = conv.exec(x, clip=True)
    yc = spi.convolve(x, kern, mode='constant', cval=0)
    print(np.round(y).astype(np.int32))
    # print(y)
    print(yc)
    assert np.all(np.round(y).astype(np.int32) == yc)
    print("---------")

    corr = RealConvolver(shape, kern, correlateInstead=True)
    y = corr.exec(x, clip=False)
    yc = spi.correlate(x, kern, mode='constant', cval=0)
    print(np.fft.fftshift(np.round(y).astype(np.int32)))
    # print(y)
    print(yc)

