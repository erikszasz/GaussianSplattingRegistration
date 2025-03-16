from PySide6.QtCore import QObject, Signal
import numpy as np


class UIStateRepository(QObject):
    signal_transformation_changed = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._transformation_matrix = np.eye(4)

    @property
    def transformation_matrix(self):
        return self._transformation_matrix

    @transformation_matrix.setter
    def transformation_matrix(self, matrix):
        if not np.array_equal(self._transformation_matrix, matrix):
            self._transformation_matrix = matrix
            self.signal_transformation_changed.emit(matrix)
