"""Regression tests for sharing one subplot's viewport through a Base64 code."""

import base64
import math
import struct
import unittest
from dataclasses import fields
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox, QPushButton

from pyimgroutines.plots import PgFigure, PgPlotItemState, ViewportStateDialog


class PgPlotItemStateTests(unittest.TestCase):
    """Check the serialized viewport format and rejection of invalid codes."""

    def test_base64_round_trip(self):
        """Required viewport bounds precede optional tagged colorbar levels."""
        state = PgPlotItemState((1.25, 2.5), (-3.75, 4.125), (0.5, 4.5))
        encoded = state.to_base64()

        self.assertEqual(
            tuple(field.name for field in fields(PgPlotItemState)),
            ("version", "xlims", "ylims", "colorbar_levels", "image_annotations"),
        )
        self.assertEqual(
            base64.urlsafe_b64decode(encoded),
            struct.pack("!I4dB2d", 2, 1.25, 2.5, -3.75, 4.125, 1, 0.5, 4.5),
        )
        self.assertEqual(len(encoded), 72)
        self.assertEqual(state.version, 2)
        self.assertEqual(PgPlotItemState.from_base64(encoded), state)

    def test_viewport_only_and_legacy_codes(self):
        """Optional fields may be absent, and existing v1 codes still apply."""
        viewport = struct.pack("!I4d", 2, 1.25, 2.5, -3.75, 4.125)
        state = PgPlotItemState((1.25, 2.5), (-3.75, 4.125))
        self.assertEqual(base64.urlsafe_b64decode(state.to_base64()), viewport)
        self.assertEqual(PgPlotItemState.from_base64(state.to_base64()), state)
        legacy = base64.urlsafe_b64encode(struct.pack("!I4d", 1, 1.25, 2.5, -3.75, 4.125))
        self.assertEqual(PgPlotItemState.from_base64(legacy.decode("ascii")), state)

    def test_unknown_fields_are_skipped(self):
        """Retired or future codes do not prevent reading recognized fields."""
        viewport = struct.pack("!I4d", 2, 1, 2, 3, 4)
        unknown = struct.pack("!BH3s", 99, 3, b"old")
        levels = struct.pack("!B2d", 1, -2, 7)
        encoded = base64.urlsafe_b64encode(viewport + unknown + levels + unknown).decode("ascii")
        self.assertEqual(
            PgPlotItemState.from_base64(encoded),
            PgPlotItemState((1, 2), (3, 4), (-2, 7)),
        )

    def test_malformed_fields_are_rejected(self):
        """Truncation, invalid lengths, duplicates, and invalid levels fail."""
        viewport = struct.pack("!I4d", 2, 1, 2, 3, 4)
        levels = struct.pack("!B2d", 1, 0, 1)
        for payload in (
            viewport + b"\x01" + struct.pack("!d", 0),
            viewport + struct.pack("!BH", 99, 3) + b"x",
            viewport + b"\x63",
            viewport + levels + levels,
            viewport + struct.pack("!B2d", 1, 2, 1),
            viewport + struct.pack("!B2d", 1, 0, math.inf),
            struct.pack("!I4d", 1, 1, 2, 3, 4) + levels,
        ):
            encoded = base64.urlsafe_b64encode(payload).decode("ascii")
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                PgPlotItemState.from_base64(encoded)

    def test_equal_colorbar_levels_are_valid(self):
        """A constant-valued image can have identical colorbar bounds."""
        state = PgPlotItemState((1, 2), (3, 4), (5, 5))
        self.assertEqual(PgPlotItemState.from_base64(state.to_base64()), state)

    def test_image_annotations_encode_point_count_and_row_col_pairs(self):
        """Code 2 uses a u16 count, then a u32 row and col for each point."""
        points = ((0, 1), (5, 6), (10, 10))
        state = PgPlotItemState((0, 10), (0, 10), image_annotations=points)
        data = base64.urlsafe_b64decode(state.to_base64())
        self.assertEqual(
            data,
            struct.pack("!I4dBH6I", 2, 0, 10, 0, 10, 2, 3, 0, 1, 5, 6, 10, 10),
        )
        self.assertEqual(len(data), 36 + 3 + 3 * 8)
        self.assertEqual(PgPlotItemState.from_base64(state.to_base64()), state)

    def test_empty_image_annotations_are_distinct_from_absent(self):
        """Code 2 with count zero explicitly requests no image annotations."""
        state = PgPlotItemState((0, 1), (0, 1), image_annotations=())
        data = base64.urlsafe_b64decode(state.to_base64())
        self.assertEqual(data[-3:], struct.pack("!BH", 2, 0))
        self.assertEqual(PgPlotItemState.from_base64(state.to_base64()), state)
        self.assertIsNone(PgPlotItemState((0, 1), (0, 1)).image_annotations)

    def test_image_annotations_coexist_with_other_fields(self):
        """Code 2 can follow code 1 and be surrounded by unknown fields."""
        prefix = struct.pack("!I4d", 2, 0, 1, 0, 1)
        unknown = struct.pack("!BH3s", 99, 3, b"old")
        levels = struct.pack("!B2d", 1, 2, 3)
        points = struct.pack("!BH2I", 2, 1, 4, 5)
        encoded = base64.urlsafe_b64encode(prefix + unknown + points + levels + unknown)
        self.assertEqual(
            PgPlotItemState.from_base64(encoded.decode("ascii")),
            PgPlotItemState((0, 1), (0, 1), (2, 3), ((4, 5),)),
        )

    def test_invalid_image_annotations_are_rejected(self):
        """Reject invalid counts, indices, truncation, and duplicate code 2."""
        for points in (((-1, 0),), ((0, 2**32),), ((1.5, 2),), ((0, 1),) * 65536):
            with self.subTest(points=points[:1], count=len(points)), self.assertRaises(ValueError):
                PgPlotItemState((0, 1), (0, 1), image_annotations=points)
        prefix = struct.pack("!I4d", 2, 0, 1, 0, 1)
        points = struct.pack("!BH2I", 2, 1, 4, 5)
        for data in (
            prefix + b"\x02",
            prefix + struct.pack("!BH", 2, 2) + struct.pack("!2I", 4, 5),
            prefix + points + points,
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                PgPlotItemState.from_base64(base64.urlsafe_b64encode(data).decode("ascii"))

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
        for version in (0, 3):
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
            second.image(np.arange(4).reshape(2, 2))
            second.annotate((0, 1))
            first.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
            second.setRange(xRange=(20, 30), yRange=(40, 50), padding=0)
            second.cbar.setLevels((1, 2))
            figure.show()
            QApplication.processEvents()
            figure.mouseMoved(second.sceneBoundingRect().center())

            first_before = first.viewRange()
            second_before = PgPlotItemState.from_plot_item(second)
            expected = PgPlotItemState((100, 110), (-5, 5), (3, 7), ((1, 0),))
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

    def test_legacy_code_leaves_existing_colorbar_levels_alone(self):
        """Applying viewport-only codes does not reset a visible colorbar."""
        figure = PgFigure()
        try:
            figure.plt.image(np.arange(4).reshape(2, 2))
            figure.plt.annotate((0, 1))
            figure.plt.cbar.setLevels((1, 2))
            legacy = struct.pack("!I4d", 1, 10, 20, 30, 40)
            state = PgPlotItemState.from_base64(
                base64.urlsafe_b64encode(legacy).decode("ascii")
            )
            state.apply_to_plot_item(figure.plt)
            self.assertEqual(figure.plt.cbar.levels(), (1, 2))
            self.assertEqual(PgPlotItemState.from_plot_item(figure.plt), PgPlotItemState(
                (10, 20), (30, 40), (1, 2), ((0, 1),),
            ))
        finally:
            figure.dispose()

    def test_plot_without_colorbar_omits_levels(self):
        """A plot with no displayed colorbar carries only viewport bounds."""
        figure = PgFigure()
        try:
            figure.plt.image(np.arange(4).reshape(2, 2), colorbar=False)
            state = PgPlotItemState.from_plot_item(figure.plt)
            self.assertIsNone(state.colorbar_levels)
            self.assertEqual(state.image_annotations, ())
            self.assertEqual(len(base64.urlsafe_b64decode(state.to_base64())), 39)
            PgPlotItemState((10, 20), (30, 40), (1, 2)).apply_to_plot_item(figure.plt)
            self.assertEqual(figure.plt.cbar.levels(), (0, 1))
        finally:
            figure.dispose()

    def test_image_annotations_replace_only_image_annotations(self):
        """Copy image points, remove old ones, and leave scatter points alone."""
        figure = PgFigure()
        try:
            plot = figure.plt
            plot.image(np.arange(144).reshape(12, 12))
            plot.annotate((10, 10))
            plot.annotate((0, 1))
            plot.annotate((5, 6))
            scatter = plot.scatter([0.5], [0.5], brush="r")
            plot.annotateScatterPoint(scatter, 0)
            state = PgPlotItemState.from_plot_item(plot)
            self.assertEqual(state.image_annotations, ((0, 1), (5, 6), (10, 10)))

            plot.removeAnnotation((5, 6))
            plot.annotate((2, 2))
            state.apply_to_plot_item(plot)
            self.assertEqual(set(plot._annotations), {
                (0, 1), (5, 6), (10, 10), (scatter, 0),
            })

            empty = PgPlotItemState((0, 1), (0, 1), image_annotations=())
            empty.apply_to_plot_item(plot)
            self.assertEqual(set(plot._annotations), {(scatter, 0)})
        finally:
            figure.dispose()

    def test_x_hotkey_annotations_are_captured(self):
        """The user's X annotations are the image indices exported by P."""
        figure = PgFigure()
        try:
            plot = figure.plt
            plot.image(np.arange(144).reshape(12, 12))
            figure.show()
            QApplication.processEvents()
            for row, col in ((0, 1), (5, 6), (10, 10)):
                figure.mouseMoved(plot.vb.mapViewToScene(QPointF(col, row)))
                QTest.keyClick(figure, Qt.Key.Key_X)
            state = PgPlotItemState.from_base64(PgPlotItemState.from_plot_item(plot).to_base64())
            self.assertEqual(state.image_annotations, ((0, 1), (5, 6), (10, 10)))
        finally:
            figure.dispose()

    def test_image_annotations_require_matching_target_image(self):
        """Reject out-of-range indices before changing a target plot."""
        figure = PgFigure()
        try:
            figure.plt.image(np.arange(4).reshape(2, 2))
            figure.plt.annotate((0, 1))
            old_range = figure.plt.viewRange()
            state = PgPlotItemState((10, 20), (30, 40), image_annotations=((5, 6),))
            with self.assertRaisesRegex(ValueError, "annotation"):
                state.apply_to_plot_item(figure.plt)
            self.assertEqual(figure.plt.viewRange(), old_range)
            self.assertIn((0, 1), figure.plt._annotations)
        finally:
            figure.dispose()

    def test_dialog_warns_if_annotations_do_not_fit_target_image(self):
        """A valid code for a different image cannot partially update the plot."""
        figure = PgFigure()
        try:
            figure.plt.image(np.arange(4).reshape(2, 2))
            figure.show()
            QApplication.processEvents()
            original = PgPlotItemState.from_plot_item(figure.plt)
            wrong_image = PgPlotItemState((10, 20), (30, 40), image_annotations=((5, 6),))
            observed = {}

            def interact():
                dialog = QApplication.activeModalWidget()
                dialog.findChild(QLineEdit).setText(wrong_image.to_base64())
                dialog.findChild(QPushButton).click()
                observed["open"] = dialog.isVisible()
                dialog.reject()

            with patch.object(QMessageBox, "warning") as warning:
                QTimer.singleShot(0, interact)
                QTest.keyClick(figure, Qt.Key.Key_P)
                warning.assert_called_once()
            self.assertTrue(observed["open"])
            self.assertEqual(PgPlotItemState.from_plot_item(figure.plt), original)
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
