from PySide6.QtCore import Qt
from typing import Iterable
import numpy as np

class KeyBuffer:
    def __init__(self, acceptedKeys: Iterable[Qt.Key]):
        self._acceptedKeys = set(acceptedKeys)
        self._buf = list()

    @property
    def acceptedKeys(self) -> set:
        return self._acceptedKeys

    def __getitem__(self, idx):
        return self._buf[idx]

    def parseKey(self, key: Qt.Key) -> Qt.Key | None:
        if key in self._acceptedKeys:
            self._buf.append(key)
            return None
        else:
            return key

    def flush(self) -> tuple[Qt.Key]:
        keys = tuple(self._buf)
        self._buf.clear()
        return keys

    def flushString(self) -> str:
        s = "".join(chr(key) for key in self._buf)
        self._buf.clear()
        return s

class KeyBufferCoordinates(KeyBuffer):
    def __init__(self):
        # Accept comma, period, all digits
        super().__init__([
            Qt.Key.Key_Comma,
            Qt.Key.Key_Period,
            Qt.Key.Key_Colon, # support ranges
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



