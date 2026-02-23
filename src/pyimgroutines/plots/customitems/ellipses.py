from PySide6.QtGui import QPainter, QPen
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6 import QtGui
import numpy as np

class EllipseItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self._pos_radii = list()
        self._pens = list()

    def addEllipse(self, pos_radii: np.ndarray, pen: QPen):
        """
        Add an ellipse to the item.

        Parameters
        ----------
        pos_radii : np.ndarray
            [x_centre, y_centre, x_radius, y_radius]

        pen : QPen
            Style of the border.
        """
        self._pos_radii.append(pos_radii)
        self._pens.append(pen)
        # Eager updates
        self.prepareGeometryChange()
        self._compute_bounds()
        self.update()

    def addCircle(self, pos_radius: np.ndarray, pen: QPen):
        """
        Add a circle to the item.

        Parameters
        ----------
        pos_radius : np.ndarray
            [x_centre, y_centre, radius]

        pen : QPen
            Style of the border.
        """
        pos_radii = np.zeros(4, pos_radius.dtype)
        pos_radii[:3] = pos_radius
        pos_radii[3] = pos_radius[2]
        self.addEllipse(pos_radii, pen)

    def _compute_bounds(self):
        all_pos_radii = np.vstack(self._pos_radii)
        minx = np.min(all_pos_radii[:,0] - all_pos_radii[:,2])
        miny = np.min(all_pos_radii[:,1] - all_pos_radii[:,3])
        maxx = np.max(all_pos_radii[:,0] + all_pos_radii[:,2])
        maxy = np.max(all_pos_radii[:,1] + all_pos_radii[:,3])
        self._rect = QRectF(minx, miny, maxx-minx, maxy-miny)
        print(self._rect)

    def boundingRect(self) -> QRectF:
        return self._rect

    def _compute_rect(self, pos_radii: np.ndarray) -> QRectF:
        # TODO: can probably compute all rects together using numpy first?
        rect = QRectF(
            pos_radii[0] - pos_radii[2],
            pos_radii[1] - pos_radii[3],
            2*pos_radii[2],
            2*pos_radii[3]
        )
        return rect

    def paint(self, p: QPainter, opt, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for ellipse, pen in zip(self._pos_radii, self._pens):
            p.setPen(pen)
            p.drawEllipse(self._compute_rect(ellipse))

