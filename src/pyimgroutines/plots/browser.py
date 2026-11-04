from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import tracemalloc

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QSignalBlocker, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .core import PgFigure

__all__ = ["FigureFactory", "PgFigureEntry", "PgFigureBrowser"]


# A factory performs any required loading and returns one fully configured figure.
FigureFactory = Callable[[], PgFigure]


@dataclass(frozen=True)
class PgFigureEntry:
    """
    Describe one lazily constructed figure shown in a PgFigureBrowser.

    Parameters
    ----------
    label : str
        Text displayed for the entry in the browser list.
    factory : FigureFactory
        Zero-argument callable that loads any required data and returns a fully
        configured PgFigure. The factory is called only when the entry is
        selected.
    """

    label: str
    factory: FigureFactory


class PgFigureBrowser(QMainWindow):
    """
    Browse lazily constructed PgFigures with one selected figure at a time.

    The browser is a separate window containing a list of figure labels. Each
    label is associated with a zero-argument factory that may load data and
    configure an entire PgFigure, including its grid, images, colormaps,
    overlays, and signal connections. The factory is not called until its entry
    is selected.

    The browser owns its figures. By default, selecting another entry disposes
    the old figure before constructing the new one. With multiFig=True, figures
    remain open when another entry is selected, and revisiting an entry reuses
    its existing figure. Closing a figure disposes it and allows its entry to
    be loaded again. User code must not reuse disposed figures or plot items.

    Factories run synchronously on the Qt GUI thread. This keeps the API general
    and allows factories to configure Qt widgets safely, but slow file reads
    will temporarily block interaction with the windows.

    The memory label reports Python-tracked allocations, not native Qt memory.

    Parameters
    ----------
    entries : Iterable[PgFigureEntry] | None
        Optional entries to place in the browser without selecting them.
    title : str
        Browser window title.
    parent : QWidget | None
        Optional Qt parent for the browser window.
    multiFig : bool
        Whether figures from multiple entries may remain open simultaneously.
    """

    # Notify callers when a new figure becomes active or the active figure is removed.
    sigCurrentFigureChanged = Signal(object)

    # Report a failed factory without allowing its exception to escape through Qt.
    sigFigureLoadFailed = Signal(int, str, object)

    def __init__(
        self,
        entries: Iterable[PgFigureEntry] | None = None,
        title: str = "PgFigure Browser",
        parent: QWidget | None = None,
        multiFig: bool = False,
    ) -> None:
        """Create an empty browser window and optionally populate its figure list."""
        # A QApplication must exist before any browser widgets are constructed.
        pg.mkQApp()
        super().__init__(parent)
        self.setWindowTitle(title)
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # Entries retain only labels and factories, never constructed figures.
        self._entries: list[PgFigureEntry] = []

        # The browser keeps the sole library-owned references to its open figures.
        self._multiFig = multiFig
        self._openFigures: dict[int, PgFigure] = {}
        self._currentIndex: int | None = None

        # The list drives lazy loading whenever the user changes its current row.
        self._listWidget = QListWidget(self)
        self._listWidget.currentRowChanged.connect(self._onCurrentRowChanged)

        # An explicit reload button also lets a failed or unchanged entry be retried.
        self._reloadButton = QPushButton("Reload selected", self)
        self._reloadButton.clicked.connect(self.reloadCurrent)
        self._memoryLabel = QLabel("Estimated memory usage: -- MiB", self)

        # Keep the browser UI deliberately small: a list, an action, and a readout.
        centralWidget = QWidget(self)
        layout = QVBoxLayout(centralWidget)
        layout.addWidget(self._listWidget)
        layout.addWidget(self._reloadButton)
        layout.addWidget(self._memoryLabel)
        self.setCentralWidget(centralWidget)
        self.statusBar().showMessage("Select a figure")

        # Populate initial entries without triggering any potentially expensive factory.
        if entries is not None:
            for entry in entries:
                self.addFigure(entry.label, entry.factory)

    @property
    def listWidget(self) -> QListWidget:
        """Return the underlying list widget for optional UI customization."""
        return self._listWidget

    @property
    def entries(self) -> tuple[PgFigureEntry, ...]:
        """Return an immutable snapshot of the registered figure entries."""
        return tuple(self._entries)

    @property
    def currentFigure(self) -> PgFigure | None:
        """Return the active figure, or None when no figure is active."""
        return self._openFigures.get(self._currentIndex) if self._currentIndex is not None else None

    @property
    def currentIndex(self) -> int | None:
        """Return the active entry index, or None when no figure is active."""
        return self._currentIndex

    def addFigure(self, label: str, factory: FigureFactory) -> int:
        """
        Add a lazy figure entry to the end of the browser list.

        Parameters
        ----------
        label : str
            Text displayed in the browser list.
        factory : FigureFactory
            Callable that returns a newly constructed and configured PgFigure.

        Returns
        -------
        int
            Zero-based index assigned to the new entry.
        """
        # Reject non-callable values when the entry is registered rather than selected.
        if not callable(factory):
            raise TypeError("factory must be callable")

        # Keep the entry model and visible list in matching row order.
        entry = PgFigureEntry(str(label), factory)
        self._entries.append(entry)
        self._listWidget.addItem(entry.label)
        return len(self._entries) - 1

    def removeFigure(self, index: int) -> None:
        """
        Remove one entry and dispose its figure if it is open.

        Parameters
        ----------
        index : int
            Zero-based index of the entry to remove.
        """
        self._validateIndex(index)

        # Block row-change callbacks while the model and widget are updated together.
        blocker = QSignalBlocker(self._listWidget)
        selectedIndex = self._listWidget.currentRow()

        # Removing an entry also removes its live figure immediately.
        self._disposeFigure(index)
        if selectedIndex == index:
            self._currentIndex = None
            selectedIndex = -1
        elif self._currentIndex is not None and index < self._currentIndex:
            # Account for the active entry moving up by one row.
            self._currentIndex -= 1

        # Remove matching entries from the Python model and visible Qt list.
        self._openFigures = {
            row - (row > index): figure for row, figure in self._openFigures.items()
        }
        self._entries.pop(index)
        removedItem = self._listWidget.takeItem(index)
        del removedItem

        # Preserve a non-active selection after rows below it have shifted.
        if selectedIndex > index:
            selectedIndex -= 1
        self._listWidget.setCurrentRow(selectedIndex)
        del blocker

    def clearFigures(self) -> None:
        """Dispose all open figures and remove every registered entry."""
        # Prevent clear() from invoking the row-change handler during teardown.
        blocker = QSignalBlocker(self._listWidget)
        for index in tuple(self._openFigures):
            self._disposeFigure(index)
        self._currentIndex = None
        self._entries.clear()
        self._listWidget.clear()
        del blocker

        # Tell the user that the browser no longer has anything to display.
        self.statusBar().showMessage("No figures")

    def selectFigure(self, index: int) -> None:
        """
        Select and lazily construct a figure by index.

        Parameters
        ----------
        index : int
            Zero-based index of the entry to select.
        """
        self._validateIndex(index)

        # Setting a different row emits currentRowChanged and performs the load.
        if self._listWidget.currentRow() != index:
            self._listWidget.setCurrentRow(index)
            return

        # A row already selected without an active figure may follow a failed load.
        if self.currentFigure is None:
            self._loadFigure(index, force=True)

    @Slot()
    def reloadCurrent(self) -> None:
        """Dispose and reconstruct the currently selected list entry."""
        selectedIndex = self._listWidget.currentRow()

        # Reload is a no-op while the list has no current selection.
        if selectedIndex < 0:
            return
        self._loadFigure(selectedIndex, force=True)

    @Slot(int)
    def _onCurrentRowChanged(self, index: int) -> None:
        """
        Load the entry selected by the list widget.

        Parameters
        ----------
        index : int
            Newly selected row, or -1 when the selection was cleared.
        """
        # Clearing the selection means that no figure should remain active.
        if index < 0:
            currentFigure = self.currentFigure
            if not self._multiFig and self._currentIndex is not None:
                self._disposeFigure(self._currentIndex)
            self._currentIndex = None
            if self._multiFig and currentFigure is not None:
                self.sigCurrentFigureChanged.emit(None)
            return
        self._loadFigure(index)

    def _loadFigure(self, index: int, force: bool = False) -> None:
        """
        Reuse an open figure or invoke an entry's factory.

        Parameters
        ----------
        index : int
            Zero-based entry index to load.
        force : bool
            Whether to reconstruct an entry that is already active.
        """
        self._validateIndex(index)

        entry = self._entries[index]
        existingFigure = self._openFigures.get(index)

        # A normal repeated selection keeps the existing figure; reload uses force.
        if not force and existingFigure is not None:
            changed = self._currentIndex != index
            self._currentIndex = index
            existingFigure.show()
            existingFigure.raise_()
            existingFigure.activateWindow()
            self.statusBar().showMessage(entry.label)
            if changed:
                self.sigCurrentFigureChanged.emit(existingFigure)
            return

        # Disable controls while processEvents paints the loading status message.
        self._listWidget.setEnabled(False)
        self._reloadButton.setEnabled(False)
        self.statusBar().showMessage(f"Loading {entry.label}...")
        QApplication.processEvents()

        # Keep a local handle so a figure can be disposed if setup fails after creation.
        figure: PgFigure | None = None
        previousFigure = self.currentFigure
        try:
            # Dispose replaced figures before invoking the factory.
            if self._multiFig:
                if force:
                    self._disposeFigure(index)
            else:
                for openIndex in tuple(self._openFigures):
                    self._disposeFigure(openIndex)
            self._currentIndex = None

            # The factory owns all loading and complete PgFigure configuration.
            factoryResult = entry.factory()
            if not isinstance(factoryResult, PgFigure):
                raise TypeError(
                    f"factory for {entry.label!r} returned {type(factoryResult).__name__}, "
                    "expected PgFigure"
                )
            figure = factoryResult

            self._openFigures[index] = figure
            figure.installEventFilter(self)
            self._currentIndex = index
            figure.show()
            self._updateMemoryLabel()

            # Publish the newly active figure only after it has been fully configured.
            self.statusBar().showMessage(entry.label)
            self.sigCurrentFigureChanged.emit(figure)
        except Exception as error:
            # A figure created before a later setup failure must not remain resident.
            wasCurrent = self._currentIndex == index
            if figure is not None:
                if self._openFigures.get(index) is figure:
                    self._disposeFigure(index)
                else:
                    figure.dispose()
            if self._multiFig and not wasCurrent and any(
                openFigure is previousFigure for openFigure in self._openFigures.values()
            ):
                self.sigCurrentFigureChanged.emit(None)

            # Keep the browser usable and expose failures through status and a signal.
            self.statusBar().showMessage(f"Failed to load {entry.label}: {error}")
            self.sigFigureLoadFailed.emit(index, entry.label, error)
        finally:
            # Re-enable navigation whether the factory succeeded or failed.
            self._listWidget.setEnabled(True)
            self._reloadButton.setEnabled(True)

    def _disposeFigure(self, index: int) -> None:
        """Immediately dispose an open figure, if one exists for the entry."""
        figure = self._openFigures.pop(index, None)
        if figure is None:
            return

        # Drop browser ownership before performing the irreversible disposal.
        wasCurrent = self._currentIndex == index
        if wasCurrent:
            self._currentIndex = None
        figure.dispose()
        self._updateMemoryLabel()
        if wasCurrent:
            self.sigCurrentFigureChanged.emit(None)

    def _updateMemoryLabel(self) -> None:
        current, _ = tracemalloc.get_traced_memory()
        self._memoryLabel.setText(f"Tracked memory: {current / 2**20:.1f} MiB")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Close and any(
            figure is watched for figure in self._openFigures.values()
        ):
            QTimer.singleShot(0, lambda: self._onFigureClosed(watched))
        return super().eventFilter(watched, event)

    def _onFigureClosed(self, figure: QObject) -> None:
        for index, openFigure in tuple(self._openFigures.items()):
            if openFigure is figure:
                wasSelected = self._listWidget.currentRow() == index
                self._disposeFigure(index)
                if wasSelected:
                    self._listWidget.setCurrentRow(-1)
                    self.statusBar().showMessage("Select a figure")
                return

    def _validateIndex(self, index: int) -> None:
        """
        Raise IndexError when an entry index is outside the list.

        Parameters
        ----------
        index : int
            Candidate zero-based entry index.
        """
        if index < 0 or index >= len(self._entries):
            raise IndexError(
                f"figure index {index} is outside the valid range "
                f"0:{len(self._entries)}"
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Dispose all open figures when the browser window closes.

        Parameters
        ----------
        event : QCloseEvent
            Qt close event delivered to the browser.
        """
        # The browser owns its open figures, so both windows share a lifetime.
        for index in tuple(self._openFigures):
            self._disposeFigure(index)
        super().closeEvent(event)
