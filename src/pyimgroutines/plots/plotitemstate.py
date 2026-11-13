"""Copy and restore a single subplot's state using a shareable code.

Version 1 uses ``!I4d``: version, xmin, xmax, ymin, ymax. Version 2 keeps
that required prefix and adds optional fields by code. Code 1 contains ``!2d``
colorbar levels directly after its one-byte code. Code 2 contains a ``!H``
point count followed by that many ``!II`` image row/column pairs. Other codes
use ``!H`` byte length followed by their payload, allowing unknown codes to
be skipped. Known codes with special framing must remain recognizable even if
retired, and field codes must not be reused. The bytes are URL-safe Base64.
Version 1 codes remain readable, without optional fields.
"""

from __future__ import annotations

import base64
import binascii
import struct
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np
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
    - colorbar_levels (added in v2): optional (minimum, maximum) colorbar levels
    - image_annotations (added in v2): optional image (row, col) indices
    """

    _COLORBAR_LEVELS_CODE = 1
    _IMAGE_ANNOTATIONS_CODE = 2

    # State fields appear in the same order as their serialized values.
    version: int = field(default=2, init=False)
    xlims: tuple[float, float]
    ylims: tuple[float, float]
    colorbar_levels: tuple[float, float] | None = None
    image_annotations: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self):
        """Reject unsupported formats and limits that cannot describe a viewport."""
        if self.version != 2:
            raise ValueError(f"Unsupported viewport state version: {self.version}.")
        if (len(self.xlims) != 2 or len(self.ylims) != 2
                or not all(isfinite(value) for value in (*self.xlims, *self.ylims))
                or self.xlims[0] >= self.xlims[1]
                or self.ylims[0] >= self.ylims[1]):
            raise ValueError("Viewport limits must be finite, increasing pairs.")
        if (self.colorbar_levels is not None
                and (len(self.colorbar_levels) != 2
                     or not all(isfinite(value) for value in self.colorbar_levels)
                     or self.colorbar_levels[0] > self.colorbar_levels[1])):
            raise ValueError("Colorbar levels must be finite, ordered pairs.")
        if (self.image_annotations is not None
                and (len(self.image_annotations) > 65535
                     or any(len(point) != 2 or any(
                         not isinstance(index, int) or isinstance(index, bool)
                         or not 0 <= index <= 0xFFFFFFFF for index in point
                     ) for point in self.image_annotations))):
            raise ValueError("Image annotations must contain at most 65535 u32 row/column pairs.")

    @classmethod
    def from_plot_item(cls, plot: PgPlotItem) -> PgPlotItemState:
        """Capture the viewport, colorbar, and image annotations in ``plot``."""
        xlims, ylims = plot.viewRange()
        levels = tuple(plot.cbar.levels()) if plot.cbar.scene() is not None else None
        annotations = None
        if plot.im is not None and plot.imgData.size:
            annotations = tuple(sorted(
                key for key in plot._annotations
                if isinstance(key[0], int) and isinstance(key[1], int)
            ))
        return cls(tuple(xlims), tuple(ylims), levels, annotations)

    def apply_to_plot_item(self, plot: PgPlotItem) -> None:
        """Set ``plot`` to these limits and optional colorbar/annotations.

        Axis limits are applied without padding. Pyqtgraph may still adjust
        them for linked axes or a locked aspect ratio.
        """
        if self.image_annotations:
            if (plot.im is None or plot.imgData.size == 0
                    or any(row >= plot.imgData.shape[0] or col >= plot.imgData.shape[1]
                           for row, col in self.image_annotations)):
                raise ValueError("Image annotation indices must fit the target image.")
        plot.setRange(xRange=self.xlims, yRange=self.ylims, padding=0)
        if self.colorbar_levels is not None and plot.cbar.scene() is not None:
            plot.cbar.setLevels(self.colorbar_levels)
        if self.image_annotations is not None:
            for key in tuple(plot._annotations):
                if isinstance(key[0], int) and isinstance(key[1], int):
                    plot.removeAnnotation(key)
            for point in self.image_annotations:
                plot.annotate(point)

    def to_base64(self) -> str:
        """Encode the required viewport and optional fields as URL-safe Base64."""
        # Viewport: version (4) + xmin/xmax/ymin/ymax (4 * 8) = 36 bytes.
        # Alone, these encode to 48 Base64 characters (48 ASCII bytes).
        data = struct.pack("!I4d", self.version, *self.xlims, *self.ylims)
        if self.colorbar_levels is not None:
            # Colorbar: code (1) + low/high (2 * 8) = 17 bytes.
            # After the 36-byte viewport: 53 bytes total -> 72 Base64 characters
            # (72 ASCII bytes), 24 more than the viewport-only code.
            data += struct.pack("!B2d", self._COLORBAR_LEVELS_CODE, *self.colorbar_levels)
        if self.image_annotations is not None:
            # Image annotations: code (1) + point count (2) + N * (row u32, col u32)
            # = 3 + 8*N bytes. Alone after the viewport, Base64 uses
            # 4 * ceil((39 + 8*N) / 3) ASCII bytes (52 bytes when N=0).
            data += struct.pack("!BH", self._IMAGE_ANNOTATIONS_CODE, len(self.image_annotations))
            data += np.asarray(self.image_annotations, dtype=">u4").tobytes()
        return base64.urlsafe_b64encode(data).decode("ascii")

    @classmethod
    def from_base64(cls, text: str) -> PgPlotItemState:
        """Decode v1 or v2 codes, skipping unrecognized optional fields."""
        try:
            data = base64.b64decode(text.strip(), altchars=b"-_", validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid Base64 viewport state.") from exc
        if len(data) == struct.calcsize("!4d"):
            raise ValueError("Unversioned viewport state codes are not supported.")
        if len(data) < struct.calcsize("!I"):
            raise ValueError("Viewport state must start with a 32-bit version.")
        (version,) = struct.unpack_from("!I", data)
        if version not in (1, 2):
            raise ValueError(f"Unsupported viewport state version: {version}.")
        header_size = struct.calcsize("!I4d")
        if len(data) < header_size:
            raise ValueError("Viewport state must contain four double-precision bounds.")
        if version == 1 and len(data) != header_size:
            raise ValueError("Version 1 viewport state must contain four double-precision numbers.")
        # Viewport: version (4) + xmin/xmax/ymin/ymax (4 * 8) = 36 bytes.
        # A viewport-only code is 48 Base64 characters (48 ASCII bytes).
        _, xmin, xmax, ymin, ymax = struct.unpack_from("!I4d", data)
        colorbar_levels = None
        image_annotations = None
        offset = header_size
        while offset < len(data):
            code = data[offset]
            offset += 1
            if code == cls._COLORBAR_LEVELS_CODE:
                # Colorbar: code (1) + low/high (2 * 8) = 17 bytes.
                # Directly after the viewport, 53 bytes total encode to
                # 72 Base64 characters (72 ASCII bytes), 24 extra.
                if colorbar_levels is not None or len(data) - offset < struct.calcsize("!2d"):
                    raise ValueError("Invalid colorbar levels field.")
                colorbar_levels = struct.unpack_from("!2d", data, offset)
                offset += struct.calcsize("!2d")
            elif code == cls._IMAGE_ANNOTATIONS_CODE:
                # Image annotations: code (1) + point count (2) + N * (row u32, col u32)
                # = 3 + 8*N bytes. Alone after the viewport, Base64 uses
                # 4 * ceil((39 + 8*N) / 3) ASCII bytes (52 bytes when N=0).
                if image_annotations is not None or len(data) - offset < struct.calcsize("!H"):
                    raise ValueError("Invalid image annotations field.")
                (count,) = struct.unpack_from("!H", data, offset)
                offset += struct.calcsize("!H")
                if count > (len(data) - offset) // struct.calcsize("!II"):
                    raise ValueError("Incomplete image annotations field.")
                image_annotations = tuple(
                    struct.unpack_from("!II", data, offset + i * struct.calcsize("!II"))
                    for i in range(count)
                )
                offset += count * struct.calcsize("!II")
            else:
                if len(data) - offset < struct.calcsize("!H"):
                    raise ValueError("Incomplete viewport state field header.")
                (length,) = struct.unpack_from("!H", data, offset)
                offset += struct.calcsize("!H")
                if length > len(data) - offset:
                    raise ValueError("Incomplete viewport state field payload.")
                offset += length
        return cls((xmin, xmax), (ymin, ymax), colorbar_levels, image_annotations)


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
            state.apply_to_plot_item(self._plot)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid viewport state", str(exc))
            return
        self.accept()
