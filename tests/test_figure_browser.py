import unittest
from unittest.mock import patch

import shiboken6
from PySide6.QtWidgets import QApplication

from pyimgroutines.plots import PgFigure, PgFigureBrowser, PgFigureEntry


class PgFigureBrowserTests(unittest.TestCase):
    def setUp(self):
        self.created = []
        self.browser = PgFigureBrowser()

    def tearDown(self):
        self.browser.close()

    def use_multi_mode(self):
        self.browser.close()
        self.browser = PgFigureBrowser(multiFig=True)

    def add_figure(self, label):
        def factory():
            figure = PgFigure(title=label)
            self.created.append(figure)
            return figure

        return self.browser.addFigure(label, factory)

    def test_default_replaces_the_previous_figure(self):
        self.add_figure("first")
        self.add_figure("second")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure

        self.browser.selectFigure(1)

        self.assertFalse(shiboken6.isValid(first))
        self.assertIs(self.browser.currentFigure, self.created[1])
        self.assertEqual(self.browser.currentIndex, 1)

    def test_default_disposes_manually_closed_figure(self):
        self.add_figure("first")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure

        first.close()
        QApplication.processEvents()

        self.assertFalse(shiboken6.isValid(first))
        self.assertIsNone(self.browser.currentFigure)
        self.assertEqual(self.browser.listWidget.currentRow(), -1)
        self.browser.listWidget.setCurrentRow(0)
        self.assertIs(self.browser.currentFigure, self.created[1])

    def test_replacement_does_not_track_geometry(self):
        self.add_figure("first")
        self.add_figure("second")
        self.browser.selectFigure(0)
        self.browser.selectFigure(1)
        self.assertFalse(hasattr(self.browser, "_figureGeometry"))
        self.assertFalse(hasattr(self.browser, "_figureWasMaximized"))

    def test_multi_mode_reuses_open_figures_and_updates_current_signal(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("second")
        changes = []
        self.browser.sigCurrentFigureChanged.connect(changes.append)

        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        self.browser.selectFigure(1)
        second = self.browser.currentFigure
        self.browser.selectFigure(0)

        self.assertEqual(len(self.created), 2)
        self.assertTrue(shiboken6.isValid(first))
        self.assertTrue(shiboken6.isValid(second))
        self.assertTrue(first.isVisible())
        self.assertTrue(second.isVisible())
        self.assertIs(self.browser.currentFigure, first)
        self.assertEqual(self.browser.currentIndex, 0)
        self.assertEqual(changes, [first, second, first])

    def test_memory_label_updates_after_open_and_reload(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("second")
        layout = self.browser.centralWidget().layout()
        memoryLabel = layout.itemAt(2).widget()
        self.assertEqual(layout.itemAt(1).widget().text(), "Reload selected")
        self.assertEqual(memoryLabel.text(), "Tracked memory: -- MiB")

        with patch(
            "pyimgroutines.plots.browser.tracemalloc.get_traced_memory",
            side_effect=[
                (12 * 2**20, 0), (20 * 2**20, 0),
                (10 * 2**20, 0), (11 * 2**20, 0),
            ],
        ) as memory:
            self.browser.selectFigure(0)
            self.assertEqual(memoryLabel.text(), "Tracked memory: 12.0 MiB")
            self.browser.selectFigure(1)
            self.assertEqual(memoryLabel.text(), "Tracked memory: 20.0 MiB")
            self.browser.selectFigure(0)
            self.assertEqual(memoryLabel.text(), "Tracked memory: 20.0 MiB")
            self.browser.reloadCurrent()
            self.assertEqual(memoryLabel.text(), "Tracked memory: 11.0 MiB")
            self.assertEqual(memory.call_count, 4)

    def test_memory_label_updates_after_manual_close(self):
        self.add_figure("first")
        memoryLabel = self.browser.centralWidget().layout().itemAt(2).widget()
        with patch(
            "pyimgroutines.plots.browser.tracemalloc.get_traced_memory",
            side_effect=[(20 * 2**20, 0), (5 * 2**20, 0)],
        ):
            self.browser.selectFigure(0)
            self.assertEqual(memoryLabel.text(), "Tracked memory: 20.0 MiB")
            self.browser.currentFigure.close()
            QApplication.processEvents()
            self.assertEqual(memoryLabel.text(), "Tracked memory: 5.0 MiB")

    def test_manual_close_disposes_and_allows_reselection(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("second")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        self.browser.selectFigure(1)
        second = self.browser.currentFigure
        second.close()
        QApplication.processEvents()

        self.assertFalse(shiboken6.isValid(second))
        self.assertTrue(shiboken6.isValid(first))
        self.assertIsNone(self.browser.currentFigure)
        self.assertIsNone(self.browser.currentIndex)
        self.assertEqual(self.browser.listWidget.currentRow(), -1)

        self.browser.listWidget.setCurrentRow(1)
        self.assertEqual(len(self.created), 3)
        self.assertIs(self.browser.currentFigure, self.created[2])

        first.close()
        QApplication.processEvents()
        self.assertFalse(shiboken6.isValid(first))
        self.assertIs(self.browser.currentFigure, self.created[2])

    def test_reload_replaces_only_selected_figure(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("second")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        self.browser.selectFigure(1)
        second = self.browser.currentFigure

        self.browser.reloadCurrent()

        self.assertTrue(shiboken6.isValid(first))
        self.assertFalse(shiboken6.isValid(second))
        self.assertIs(self.browser.currentFigure, self.created[2])

    def test_remove_reindexes_open_figures_and_disposes_removed_one(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("middle")
        self.add_figure("last")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        self.browser.selectFigure(2)
        last = self.browser.currentFigure

        self.browser.removeFigure(0)

        self.assertFalse(shiboken6.isValid(first))
        self.assertTrue(shiboken6.isValid(last))
        self.assertEqual(self.browser.currentIndex, 1)
        self.assertIs(self.browser.currentFigure, last)
        self.browser.selectFigure(1)
        self.assertEqual(len(self.created), 2)

        self.browser.removeFigure(1)
        self.assertFalse(shiboken6.isValid(last))
        self.assertIsNone(self.browser.currentFigure)
        self.assertIsNone(self.browser.currentIndex)

    def test_failed_load_does_not_dispose_other_open_figures(self):
        self.use_multi_mode()
        self.add_figure("first")

        def fail():
            raise RuntimeError("failed")

        self.browser.addFigure("broken", fail)
        changes = []
        failures = []
        self.browser.sigCurrentFigureChanged.connect(changes.append)
        self.browser.sigFigureLoadFailed.connect(lambda *args: failures.append(args))
        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        self.browser.selectFigure(1)

        self.assertTrue(shiboken6.isValid(first))
        self.assertIsNone(self.browser.currentFigure)
        self.assertIsNone(self.browser.currentIndex)
        self.assertEqual(changes, [first, None])
        self.assertEqual(failures[0][:2], (1, "broken"))

        self.browser.selectFigure(0)
        self.assertIs(self.browser.currentFigure, first)
        self.assertEqual(len(self.created), 1)

    def test_clear_and_browser_close_dispose_every_open_figure(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("second")
        self.browser.selectFigure(0)
        self.browser.selectFigure(1)

        self.browser.clearFigures()
        self.assertFalse(any(shiboken6.isValid(figure) for figure in self.created))
        self.assertEqual(self.browser.entries, ())

        self.add_figure("third")
        self.browser.selectFigure(0)
        last = self.browser.currentFigure
        self.browser.close()
        self.assertFalse(shiboken6.isValid(last))

    def test_clearing_selection_keeps_open_figures_in_multi_mode(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        changes = []
        self.browser.sigCurrentFigureChanged.connect(changes.append)

        self.browser.listWidget.setCurrentRow(-1)

        self.assertIsNone(self.browser.currentFigure)
        self.assertIsNone(self.browser.currentIndex)
        self.assertTrue(shiboken6.isValid(first))
        self.browser.selectFigure(0)
        self.assertIs(self.browser.currentFigure, first)
        self.assertEqual(changes, [None, first])

    def test_pending_close_does_not_dispose_reindexed_figure(self):
        self.use_multi_mode()
        self.add_figure("first")
        self.add_figure("second")
        self.browser.selectFigure(0)
        first = self.browser.currentFigure
        self.browser.selectFigure(1)
        second = self.browser.currentFigure

        first.close()
        self.browser.removeFigure(0)
        QApplication.processEvents()

        self.assertFalse(shiboken6.isValid(first))
        self.assertTrue(shiboken6.isValid(second))
        self.assertIs(self.browser.currentFigure, second)


if __name__ == "__main__":
    unittest.main()
