from PySide6.QtWidgets import QGraphicsPolygonItem
from PySide6.QtGui import QPolygonF, QPen, QBrush, QColor
from PySide6.QtCore import QPointF, Qt
import pyqtgraph as pg
import numpy as np

class HoverPolygonItem(QGraphicsPolygonItem):
    """
    A hoverable polygon item that changes appearance when the mouse is over it.
    """
    def __init__(
        self,
        vertices: np.ndarray | list,
        pen: QPen = pg.mkPen("w"),
        brush: QBrush = QBrush(QColor(0, 0, 0, 0)),
        hoverBrush: QBrush = QBrush(QColor(0, 0, 0, 50)),
        zValue: float = 100
    ):
        """
        A polygon item that changes appearance on hover.

        Parameters
        ----------
        vertices : np.ndarray | list
            Polygon vertices as Nx2 array or list of (x, y) tuples.
            Do not repeat the first vertex at the end.

        pen : QPen
            Border style. Defaults to white.

        brush : QBrush
            Fill color (normal state). Defaults to fully transparent black.

        hoverBrush : QBrush
            Fill color when hovered. Defaults to slightly visible black.

        zValue : float
            Z-order of the item. Higher values are drawn on top. Defaults to 100.
        """
        polygon = QPolygonF([QPointF(x, y) for x, y in vertices])
        super().__init__(polygon)

        self._normalBrush = brush
        self._hoverBrush = hoverBrush

        self.setPen(pen)
        self.setBrush(self._normalBrush)
        self.setZValue(zValue)
        self.setAcceptHoverEvents(True)
        self._isHovered = False

    def _isInsidePolygon(self, pos: QPointF) -> bool:
        return self.polygon().containsPoint(pos, Qt.FillRule.WindingFill)

    def hoverMoveEvent(self, ev):
        # NOTE: we don't use hoverEnterEvent because that relies on the bounding rect,
        # and won't really update based on arbitrary polygons correctly
        # hence we only toggle _isHovered when we are actually inside the polygon here
        inside = self._isInsidePolygon(ev.pos())
        # Check _isHovered to avoid redundant setBrush/update calls on every mouse move
        if inside and not self._isHovered:
            self._isHovered = True
            self.setBrush(self._hoverBrush)
            self.update()
        elif not inside and self._isHovered:
            self._isHovered = False
            self.setBrush(self._normalBrush)
            self.update()
        super().hoverMoveEvent(ev)

    def hoverLeaveEvent(self, ev):
        if self._isHovered:
            self._isHovered = False
            self.setBrush(self._normalBrush)
            self.update()
        super().hoverLeaveEvent(ev)


class ClickPolygonItem(HoverPolygonItem):
    """
    A clickable polygon item that triggers onClicked when clicked inside.
    Subclass this and reimplement onClicked to handle click events.
    """
    def onClicked(self, ev):
        """
        Called when the polygon is clicked inside its bounds.
        Reimplement this method in subclasses to handle click events.
        """
        print(ev.pos())

    def mousePressEvent(self, ev):
        if self._isInsidePolygon(ev.pos()):
            self.onClicked(ev)
            ev.accept()
        else:
            super().mousePressEvent(ev)


if __name__ == "__main__":
    # from pyimgroutines.plots import PgFigure, forceShow
    from ..core import PgFigure, forceShow

    fig = PgFigure()
    fig.plt.image(np.arange(4).reshape(2,2), [-0.5,0,2,1.5], addHalfPixelBorder=False)
    vertices = [(0, 0), (1, 0), (1.5, 1), (0.5, 1.5), (-0.5, 0.5)]
    polygon = ClickPolygonItem(vertices)
    fig.plt.addItem(polygon)
    fig.plt.setAspectLocked()
    fig.show()
    forceShow()
