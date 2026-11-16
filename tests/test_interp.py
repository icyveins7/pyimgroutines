import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import map_coordinates

from pyimgroutines.interp import GridBilerp


class GridBilerpTests(unittest.TestCase):
    def test_numpy_grid_matches_scalar_for_zero_and_nan_outside(self):
        image = np.array([[1000, 0, 5], [2000, 1000, 10], [0, 3000, 7]], dtype=np.uint16)
        x = np.array([[-3.0, -0.2, 0.0, 0.5, 1.7], [2.0, 2.4, 3.1, 0.25, 1.3]])
        y = np.array([[0.5, 0.5, 0.0, 1.5, 2.0], [2.0, 1.5, 1.0, -0.5, 3.5]])

        for nan_outside in (False, True):
            with self.subTest(nan_outside=nan_outside):
                interpolator = GridBilerp(image, nanOutside=nan_outside)
                expected = np.array([
                    interpolator._bilerp(float(px), float(py))
                    for px, py in zip(x.flat, y.flat)
                ]).reshape(x.shape)

                np.testing.assert_allclose(interpolator._bilerpNumpy(x, y), expected, equal_nan=True)
                with patch.object(interpolator, "_bilerp", side_effect=AssertionError("scalar path used")):
                    np.testing.assert_allclose(interpolator.exec(x, y), expected, equal_nan=True)

    def test_numpy_grid_respects_origin_and_steps(self):
        image = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)
        interpolator = GridBilerp(image, x0=10.0, xstep=2.0, y0=-4.0, ystep=0.5, nanOutside=True)
        x, y = np.meshgrid(np.linspace(9.0, 13.0, 13), np.linspace(-4.25, -3.25, 11))

        result = interpolator._bilerpNumpy(x, y)
        expected = map_coordinates(
            image, [(y + 4.0) / 0.5, (x - 10.0) / 2.0],
            order=1, mode="constant", cval=np.nan, prefilter=False, output=np.float64,
        )

        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_uint16_subtraction_does_not_wrap(self):
        image = np.array([[1000, 0], [1000, 0]], dtype=np.uint16)

        result = GridBilerp(image).exec(np.array([0.5]), np.array([0.5]))

        self.assertEqual(result[0], 500.0)

    def test_nan_outside_matches_bilinear_reference_at_edges(self):
        image = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)
        x = np.array([-0.2, 0.5, 1.0, 0.0, 1.1, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5, 1.0, 0.5, -0.3, 1.2])

        actual = GridBilerp(image, nanOutside=True).exec(x, y)
        expected = map_coordinates(
            image, [y, x], order=1, mode="constant", cval=np.nan,
            prefilter=False, output=np.float64,
        )

        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_default_zero_outside_is_unchanged(self):
        image = np.array([[1000, 0], [1000, 0]], dtype=np.uint16)

        result = GridBilerp(image).exec(np.array([-0.2]), np.array([0.5]))

        self.assertEqual(result[0], 800.0)


if __name__ == "__main__":
    unittest.main()
