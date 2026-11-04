import unittest

import numpy as np
import pyqtgraph as pg

from pyimgroutines.plots import PgFigure


class ImageValidLevelsTests(unittest.TestCase):
    def setUp(self):
        self.figure = PgFigure()

    def tearDown(self):
        self.figure.dispose()

    def test_int32_sentinel_is_transparent_and_excluded_from_auto_levels(self):
        sentinel = np.iinfo(np.int32).max
        data = np.array([[0, 1000, 2000, sentinel]], dtype=np.int32)

        self.figure.plt.image(data, validLevels=(np.iinfo(np.int32).min, sentinel - 1))
        image = self.figure.plt.im
        image.render()

        self.assertIs(self.figure.plt.imgData, data)
        self.assertEqual(image.image.dtype, np.int32)
        self.assertTrue(np.shares_memory(image.image, data))
        np.testing.assert_array_equal(image.image, data)
        self.assertEqual(data[0, 3], sentinel)
        self.assertEqual(tuple(image.levels), (0, 2000))
        self.assertEqual(tuple(self.figure.plt.cbar.levels()), (0, 2000))
        self.assertEqual([image.qimage.pixelColor(x, 0).alpha() for x in range(4)], [255, 255, 255, 0])
        reference = pg.ImageItem(
            data[:, :3], axisOrder="row-major", levels=(0, 2000)
        )
        reference.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(nPts=image.lut.shape[0])
        )
        reference.render()
        self.assertEqual(
            [image.qimage.pixelColor(x, 0) for x in range(3)],
            [reference.qimage.pixelColor(x, 0) for x in range(3)],
        )

        self.figure.plt.cbar.setLevels((500, 1000))
        image.render()
        self.assertEqual(image.qimage.pixelColor(3, 0).alpha(), 0)

    def test_out_of_range_pixels_are_transparent_with_explicit_levels(self):
        data = np.array([[-1, 0, 1000, 2000, 2001]], dtype=np.int32)

        self.figure.plt.image(data, validLevels=(0, 2000), levels=(0, 2000), colorbar=False)
        image = self.figure.plt.im
        image.render()

        self.assertEqual(image.image.dtype, np.int32)
        self.assertTrue(np.shares_memory(image.image, data))
        self.assertEqual(
            [image.qimage.pixelColor(x, 0).alpha() for x in range(5)],
            [0, 255, 255, 255, 0],
        )

    def test_read_only_int16_image_data_remains_unchanged(self):
        sentinel = np.iinfo(np.int16).max
        data = np.array([[-10, 2000, sentinel]], dtype=np.int16)
        data.flags.writeable = False

        self.figure.plt.image(data, validLevels=(np.iinfo(np.int16).min, sentinel - 1))
        image = self.figure.plt.im
        image.render()

        self.assertEqual(image.image.dtype, np.int16)
        self.assertTrue(np.shares_memory(image.image, data))
        np.testing.assert_array_equal(image.image, data)
        self.assertEqual(tuple(image.levels), (-10, 2000))
        self.assertEqual([image.qimage.pixelColor(x, 0).alpha() for x in range(3)], [255, 255, 0])

    def test_auto_levels_ignore_values_below_and_above_valid_range(self):
        data = np.array([[-1, 0, 2000, 2001]], dtype=np.int32)

        self.figure.plt.image(data, validLevels=(0, 2000))

        self.assertEqual(tuple(self.figure.plt.im.levels), (0, 2000))

    def test_all_valid_pixels_do_not_require_an_image_copy(self):
        data = np.array([[0, 1000, 2000]], dtype=np.int32)

        self.figure.plt.image(data, validLevels=(0, 2000))

        self.assertTrue(np.shares_memory(self.figure.plt.im.image, data))

    def test_all_invalid_data_requires_explicit_levels(self):
        sentinel = np.iinfo(np.int32).max
        data = np.full((2, 2), sentinel, dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "valid"):
            self.figure.plt.image(data, validLevels=(0, sentinel - 1))

        self.figure.plt.image(data, validLevels=(0, sentinel - 1), levels=(0, 1))
        image = self.figure.plt.im
        image.render()
        self.assertEqual(image.qimage.pixelColor(0, 0).alpha(), 0)

    def test_valid_levels_must_be_ordered(self):
        with self.assertRaisesRegex(ValueError, "validLevels"):
            self.figure.plt.image(np.array([[1]], dtype=np.int32), validLevels=(2, 0))


if __name__ == "__main__":
    unittest.main()
