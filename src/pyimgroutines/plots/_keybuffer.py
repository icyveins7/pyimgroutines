from PySide6.QtCore import Qt, QObject, Signal
from typing import Iterable
import numpy as np

class KeyBuffer(QObject):
    bufferChanged = Signal()
    
    def __init__(self, acceptedKeys: Iterable[Qt.Key]):
        super().__init__()
        self._acceptedKeys = set(acceptedKeys)
        self._buf = list()

    @property
    def buffer(self) -> list:
        return self._buf

    @property
    def string(self) -> str:
        return "".join(chr(key) for key in self._buf)

    @property
    def acceptedKeys(self) -> set:
        return self._acceptedKeys

    def __getitem__(self, idx):
        return self._buf[idx]

    def parseKey(self, key: Qt.Key) -> Qt.Key | None:
        """
        Parse a key to see if it is in the accepted keys,
        otherwise returns None.

        Emits bufferChanged signal when a key is accepted.
        """
        # Remove keys; backspace is special and not explicitly in the acceptedKeys
        if key == Qt.Key.Key_Backspace and len(self._buf):
            self._buf.pop()
            self.bufferChanged.emit()
        elif key in self._acceptedKeys:
            self._buf.append(key)
            self.bufferChanged.emit()
            return None
        else:
            return key

    def flush(self) -> tuple[Qt.Key]:
        keys = tuple(self._buf)
        self._buf.clear()
        self.bufferChanged.emit()
        return keys

    def flushString(self) -> str:
        s = "".join(chr(key) for key in self._buf)
        self._buf.clear()
        self.bufferChanged.emit()
        return s

class KeyBufferCoordinates(KeyBuffer):
    def __init__(self):
        # Accept comma, period, minus, all digits
        super().__init__([
            Qt.Key.Key_Comma,
            Qt.Key.Key_Period,
            Qt.Key.Key_Colon, # support ranges
            Qt.Key.Key_Minus, # support negative numbers
            Qt.Key.Key_Less,
            Qt.Key.Key_Greater,
            Qt.Key.Key_Equal,
            Qt.Key.Key_E, # support scientific notation
            Qt.Key.Key_0,
            Qt.Key.Key_1,
            Qt.Key.Key_2,
            Qt.Key.Key_3,
            Qt.Key.Key_4,
            Qt.Key.Key_5,
            Qt.Key.Key_6,
            Qt.Key.Key_7,
            Qt.Key.Key_8,
            Qt.Key.Key_9,
        ])
        # Primarily a flag to indicate if the buffer is frozen
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self):
        """Freeze the buffer. Use this to help internally mark that the buffer's contents is to be used next."""
        self._frozen = True

    def unfreeze(self):
        """Unfreeze the buffer."""
        self._frozen = False

    def flushRange(self) -> tuple[float, float] | None:
        rangeString = self.flushString()
        lowerStr, upperStr = rangeString.split(":")
        lower = None if len(lowerStr) == 0 else float(lowerStr)
        upper = None if len(upperStr) == 0 else float(upperStr)
        return lower, upper

    def flushCoordinates(self) -> tuple[tuple[float,float]|float|None, tuple[float,float]|float|None]:
        coordx = None
        coordy = None
        coordString = self.flushString()
        splitcoordString = coordString.split(",")
        # Expect x,y
        if len(splitcoordString) != 2:
            # Invalid coords
            return np.nan, np.nan
        else:
            strcoordx, strcoordy = splitcoordString
            coordx = self._processCoordinate(strcoordx)
            coordy = self._processCoordinate(strcoordy)
        return coordx, coordy

    def _processCoordinate(self, coordString: str) -> None | float | tuple[float, float]:
        # Check if it's empty
        if len(coordString) == 0:
            return None

        # Check if it's a range (:)
        rangeString = coordString.split(":")
        if len(rangeString) == 2:
            return float(rangeString[0]), float(rangeString[1])

        # Or a point
        elif len(rangeString) == 1:
            return float(rangeString[0])

        # Otherwise invalid
        else:
            return np.nan



