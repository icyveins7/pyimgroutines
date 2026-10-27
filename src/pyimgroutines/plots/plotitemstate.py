"""Copy and restore a single subplot's state using a shareable code.

Version 1 uses a big-endian 32-bit unsigned integer followed by four big-endian
doubles (``!I4d``): version, xmin, xmax, ymin, ymax. The bytes are URL-safe
Base64 encoded. Both users must use the same state format version: this v1
reader accepts only v1 codes. Unversioned and other-version codes are rejected.

When changing the captured state, increment the format version, update plot
capture and application, and record the version in which each new field was
added in the class docstring. No cross-version decoding is required.
"""

from __future__ import annotations

import base64
import binascii
import struct
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING

from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

if TYPE_CHECKING:
    from .core import PgFigure, PgPlotItem


@dataclass(frozen=True)
class PgPlotItemState:
    """State of a plot item intended for replication across users.

    The payload format version is stored as a 32-bit unsigned integer.

    Currently captured:
    - xlims (added in v1): (minimum, maximum) x-axis limits
    - ylims (added in v1): (minimum, maximum) y-axis limits
    """

    # State fields appear in the same order as their serialized values.
    version: int = field(default=1, init=False)
    xlims: tuple[float, float]
    ylims: tuple[float, float]

    def __post_init__(self):
        """Reject unsupported formats and limits that cannot describe a viewport."""
        if self.version != 1:
            raise ValueError(f"Unsupported viewport state version: {self.version}.")
        if (len(self.xlims) != 2 or len(self.ylims) != 2
                or not all(isfinite(value) for value in (*self.xlims, *self.ylims))
                or self.xlims[0] >= self.xlims[1]
                or self.ylims[0] >= self.ylims[1]):
            raise ValueError("Viewport limits must be finite, increasing pairs.")

    @classmethod
    def from_plot_item(cls, plot: PgPlotItem) -> PgPlotItemState:
        """Capture the axis limits currently visible in ``plot``."""
        xlims, ylims = plot.viewRange()
        return cls(tuple(xlims), tuple(ylims))

    def apply_to_plot_item(self, plot: PgPlotItem) -> None:
        """Set ``plot`` to these limits without adding range padding.

        Pyqtgraph may still adjust the result for linked axes or a locked
        aspect ratio.
        """
        plot.setRange(xRange=self.xlims, yRange=self.ylims, padding=0)

    def to_base64(self) -> str:
        """Encode the version and four limits as copyable, URL-safe Base64."""
        data = struct.pack("!I4d", self.version, *self.xlims, *self.ylims)
        return base64.urlsafe_b64encode(data).decode("ascii")

    @classmethod
    def from_base64(cls, text: str) -> PgPlotItemState:
        """Decode a v1 code; reject other versions and unusable limits."""
        try:
            data = base64.b64decode(text.strip(), altchars=b"-_", validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid Base64 viewport state.") from exc
        if len(data) == struct.calcsize("!4d"):
            raise ValueError("Unversioned viewport state codes are not supported.")
        if len(data) < struct.calcsize("!I"):
            raise ValueError("Viewport state must start with a 32-bit version.")
        (version,) = struct.unpack_from("!I", data)
        if version != 1:
            raise ValueError(f"Viewport state version {version} does not match version 1.")
        if len(data) != struct.calcsize("!I4d"):
            raise ValueError("Version 1 viewport state must contain four double-precision numbers.")
        _, xmin, xmax, ymin, ymax = struct.unpack("!I4d", data)
        return cls((xmin, xmax), (ymin, ymax))


class ViewportStateDialog(QDialog):
    """Let a user copy one plot's viewport code or paste another to apply it.

    The figure supplies the currently hovered plot when the P hotkey is pressed.
    The dialog intentionally has only a text field and a Set button.
    """

    def __init__(self, plot: PgPlotItem, parent: PgFigure):
        """Prefill the dialog with ``plot``'s current viewport code."""
        super().__init__(parent)
        self._plot = plot
        self.setWindowTitle("Viewport state")
        layout = QVBoxLayout(self)
        self.lineEdit = QLineEdit(PgPlotItemState.from_plot_item(plot).to_base64())
        self.lineEdit.setMinimumWidth(400)
        self.lineEdit.selectAll()
        layout.addWidget(self.lineEdit)
        button = QPushButton("Set")
        button.clicked.connect(self._setState)
        layout.addWidget(button)

    def showEvent(self, event):
        """Center on the cursor's screen, not on the parent figure window."""
        super().showEvent(event)
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _setState(self):
        """Apply a valid pasted code; keep the dialog open on invalid input."""
        try:
            state = PgPlotItemState.from_base64(self.lineEdit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid viewport state", str(exc))
            return
        state.apply_to_plot_item(self._plot)
        self.accept()
