"""Regression tests for sharing one subplot's viewport through a Base64 code."""

import base64
import math
import struct
import unittest
from dataclasses import fields
from unittest.mock import patch

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox, QPushButton

from pyimgroutines.plots import PgFigure, PgPlotItemState, ViewportStateDialog


class PgPlotItemStateTests(unittest.TestCase):
    """Check the serialized viewport format and rejection of invalid codes."""

    def test_base64_round_trip(self):
        """State field order matches the v1 header and four serialized doubles."""
        state = PgPlotItemState((1.25, 2.5), (-3.75, 4.125))
        encoded = state.to_base64()

        self.assertEqual(
            tuple(field.name for field in fields(PgPlotItemState)),
            ("version", "xlims", "ylims"),
        )
        self.assertEqual(
            base64.urlsafe_b64decode(encoded),
            struct.pack("!I4d", 1, 1.25, 2.5, -3.75, 4.125),
        )
        self.assertEqual(state.version, 1)
        self.assertEqual(PgPlotItemState.from_base64(encoded), state)

    def test_unversioned_code_is_rejected(self):
        """A code without a version cannot be shared with a v1 reader."""
        unversioned = base64.urlsafe_b64encode(
            struct.pack("!4d", 1.25, 2.5, -3.75, 4.125)
        ).decode("ascii")

        with self.assertRaisesRegex(ValueError, "Unversioned"):
            PgPlotItemState.from_base64(unversioned)

    def test_invalid_payloads_are_rejected(self):
        """Malformed, truncated, reversed, and non-finite codes cannot be applied."""
        for encoded in (
            "not-base64!",
            base64.urlsafe_b64encode(b"too short").decode("ascii"),
            base64.urlsafe_b64encode(struct.pack("!I", 1) + b"too short").decode("ascii"),
            base64.urlsafe_b64encode(struct.pack("!I4d", 1, 2, 1, 0, 1)).decode("ascii"),
            base64.urlsafe_b64encode(struct.pack("!I4d", 1, 0, 1, math.nan, 1)).decode("ascii"),
        ):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                PgPlotItemState.from_base64(encoded)

    def test_unknown_version_is_rejected(self):
        """A reader never guesses how to interpret another version's layout."""
        for version in (0, 2):
            encoded = base64.urlsafe_b64encode(
                struct.pack("!I4d", version, 0, 1, 2, 3)
            ).decode("ascii")
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "version"):
                PgPlotItemState.from_base64(encoded)


class ViewportDialogTests(unittest.TestCase):
    """Exercise the P hotkey and dialog with real Qt widgets offscreen."""

    def test_p_opens_hovered_subplot_state_and_set_applies_it(self):
        """P shows the hovered plot's code at screen center; Set updates only it."""
        figure = PgFigure()
        try:
            figure.setPlotGrid(1, 2)
            first, second = figure[0, 0], figure[0, 1]
            first.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
            second.setRange(xRange=(20, 30), yRange=(40, 50), padding=0)
            figure.show()
            QApplication.processEvents()
            figure.mouseMoved(second.sceneBoundingRect().center())

            first_before = first.viewRange()
            second_before = PgPlotItemState.from_plot_item(second)
            expected = PgPlotItemState((100, 110), (-5, 5))
            observed = {}

            def interact():
                """Inspect the modal dialog and submit a new code during its event loop."""
                dialog = QApplication.activeModalWidget()
                observed["dialog"] = isinstance(dialog, ViewportStateDialog)
                if dialog is None:
                    return
                line_edit = dialog.findChild(QLineEdit)
                button = dialog.findChild(QPushButton)
                observed["widgets"] = (len(dialog.findChildren(QLineEdit)),
                                       len(dialog.findChildren(QPushButton)))
                observed["button"] = button.text()
                observed["initial"] = PgPlotItemState.from_base64(line_edit.text())
                screen = QApplication.screenAt(figure.cursor().pos()) or QApplication.primaryScreen()
                observed["center"] = dialog.frameGeometry().center() == screen.availableGeometry().center()
                line_edit.setText(expected.to_base64())
                button.click()

            QTimer.singleShot(0, interact)
            QTest.keyClick(figure, Qt.Key.Key_P)

            self.assertTrue(observed["dialog"])
            self.assertEqual(observed["widgets"], (1, 1))
            self.assertEqual(observed["button"], "Set")
            self.assertEqual(observed["initial"], second_before)
            self.assertTrue(observed["center"])
            self.assertEqual(PgPlotItemState.from_plot_item(second), expected)
            self.assertEqual(first.viewRange(), first_before)
        finally:
            figure.dispose()

    def test_invalid_code_does_not_change_plot_or_close_dialog(self):
        """An invalid pasted code warns without closing or changing the viewport."""
        figure = PgFigure()
        try:
            figure.show()
            QApplication.processEvents()
            original = PgPlotItemState.from_plot_item(figure.plt)
            observed = {}

            def interact():
                """Submit an invalid code while the modal dialog is running."""
                dialog = QApplication.activeModalWidget()
                line_edit = dialog.findChild(QLineEdit)
                line_edit.setText("bad code")
                dialog.findChild(QPushButton).click()
                observed["visible"] = dialog.isVisible()
                observed["state"] = PgPlotItemState.from_plot_item(figure.plt)
                dialog.reject()

            with patch.object(QMessageBox, "warning") as warning:
                QTimer.singleShot(0, interact)
                QTest.keyClick(figure, Qt.Key.Key_P)
                warning.assert_called_once()
            self.assertTrue(observed["visible"])
            self.assertEqual(observed["state"], original)
        finally:
            figure.dispose()


if __name__ == "__main__":
    unittest.main()
