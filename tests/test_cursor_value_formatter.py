import unittest

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pyimgroutines.plots import PgFigure


class CursorValueFormatterTests(unittest.TestCase):
    def setUp(self):
        self.figure = PgFigure()

    def tearDown(self):
        self.figure.dispose()

    def test_custom_formatter_only_affects_value_mode(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        self.figure.plt.image(data, addHalfPixelBorder=False)
        self.figure.show()
        QApplication.processEvents()
        plot = self.figure.plt
        plot._cursorPos[:] = (2.5, 1.5)
        plot._setMouseLabelTextAndPos()
        self.assertEqual(plot.mouseLabel.textItem.toPlainText(), "2.5, 1.5")

        calls = []

        def formatter(row, col, value):
            calls.append((row, col, value))
            return f"scaled: {value * 2}"

        plot.setCursorValueFormatter(formatter)
        self.assertEqual(calls, [])
        QTest.keyClick(self.figure, Qt.Key.Key_V)
        self.assertEqual(plot.mouseLabel.textItem.toPlainText(), "scaled: 12")
        self.assertEqual(calls, [(1, 2, data[1, 2])])
        np.testing.assert_array_equal(data, [[1, 2, 3], [4, 5, 6]])

        QTest.keyClick(self.figure, Qt.Key.Key_V)
        self.assertEqual(plot.mouseLabel.textItem.toPlainText(), "")
        self.assertEqual(len(calls), 1)

        QTest.keyClick(self.figure, Qt.Key.Key_V)
        self.assertEqual(plot.mouseLabel.textItem.toPlainText(), "2.5, 1.5")
        QTest.keyClick(self.figure, Qt.Key.Key_V)
        plot.setCursorValueFormatter(None)
        self.assertEqual(plot.mouseLabel.textItem.toPlainText(), "[1, 2]: 6")

    def test_out_of_bounds_does_not_call_formatter(self):
        self.figure.plt.image(np.array([[3]], dtype=np.int32))
        self.figure.show()
        QApplication.processEvents()
        plot = self.figure.plt
        plot._cursorPos[:] = (50, 50)
        calls = []
        plot.setCursorValueFormatter(lambda row, col, value: calls.append((row, col, value)) or "x")

        QTest.keyClick(self.figure, Qt.Key.Key_V)

        self.assertEqual(plot.mouseLabel.textItem.toPlainText(), "OOB")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
