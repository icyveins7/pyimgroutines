from PySide6.QtWidgets import QGraphicsProxyWidget
from pyqtgraph.graphicsItems.LegendItem import LegendItem
from pyqtgraph.widgets.ColorButton import ColorButton

from PySide6 import QtCore

class RecolorableLegendItem(LegendItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add the placeholder list of colorbuttons?
        self._colorbtns = []

    def _addItemToLayout(self, sample, label):
        # We override the entire thing because we need to adjust the layout
        col = self.layout.columnCount()
        row = self.layout.rowCount()
        if row:
            row -= 1
        nCol = self.columnCount * 3 # NOTE: New: changed to 3 here because we put the button behind
        # FIRST ROW FULL
        if col == nCol:
            for col in range(0, nCol, 2):
                # FIND RIGHT COLUMN
                if not self.layout.itemAt(row, col):
                    break
            else:
                if col + 2 == nCol:
                    # MAKE NEW ROW
                    col = 0
                    row += 1
        self.layout.addItem(sample, row, col, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.layout.addItem(label, row, col + 1)
        # Keep rowCount in sync with the number of rows if items are added
        self.rowCount = max(self.rowCount, row + 1)

        # NOTE: NEW: Now we add the button
        # we use a proxy widget in order to get it to work
        # but this makes it so it doesn't have .width(), which we handle below
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(ColorButton())
        self._colorbtns.append(proxy)
        self.layout.addItem(proxy, row, col + 2)


    def updateSize(self):
        # Essentially same as original, with 1 change below
        if self.size is not None:
            return
        height = 0
        width = 0
        for row in range(self.layout.rowCount()):
            row_height = 0
            col_width = 0
            for col in range(self.layout.columnCount()):
                item = self.layout.itemAt(row, col)
                # NOTE: ignore the colorbutton because it doesn't have .width
                if item and col % 3 != 2:
                    col_width += item.width() + 3
                    row_height = max(row_height, item.height())
            width = max(width, col_width)
            height += row_height
        self.setGeometry(0, 0, width, height)
        return
