import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QGuiApplication
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGridLayout, QLineEdit

from models.ui_state_repository import UIStateRepository
from src.gui.widgets.custom_push_button import CustomPushButton


class Transformation3DPicker(QWidget):
    class MatrixCell(QLineEdit):
        value_changed = QtCore.Signal(int, int, float)
        matrix_pasted = QtCore.Signal(np.ndarray)

        def __init__(self, row, col, value=0.0):
            super().__init__()
            self.row = row
            self.col = col

            self.setFixedSize(50, 50)
            self.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.setText(str(value))
            self.setValidator(QDoubleValidator(-9999.0, 9999.0, 10))

            self.textEdited.connect(self.update_cell_value)

        def update_cell_value(self, text):
            try:
                value = float(text)
                self.value_changed.emit(self.row, self.col, value)
            except ValueError:
                pass

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_V and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                clipboard = QGuiApplication.clipboard()
                text_original = clipboard.text()
                try:
                    text = text_original.replace("\n", "").replace("[", "").replace("]", "")
                    new_matrix = np.fromstring(text, dtype=np.float32, sep=',')
                    self.matrix_pasted.emit(new_matrix.reshape(4, 4))
                except ValueError:
                    pass

            super().keyPressEvent(event)

    def __init__(self, ui_repository: UIStateRepository):
        super().__init__()
        self.ui_repository = ui_repository

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label = QLabel("Transformation matrix")
        label.setStyleSheet(
            """QLabel {
                font-size: 12pt;
                font-weight: bold;
                padding-bottom: 0.5em;
            }"""
        )

        self.matrix_widget = QWidget()
        grid_layout = QGridLayout(self.matrix_widget)
        self.cells = []

        transformation_matrix = self.ui_repository.transformation_matrix
        for iRow in range(4):
            row_cells = []
            for iCol in range(4):
                cell = self.MatrixCell(iRow, iCol, transformation_matrix[iRow, iCol])
                cell.value_changed.connect(self.cell_value_changed)
                cell.matrix_pasted.connect(self.set_transformation)
                grid_layout.addWidget(cell, iRow, iCol)
                row_cells.append(cell)
            self.cells.append(row_cells)

        button_reset = CustomPushButton("Reset transformation matrix", 90)
        button_reset.connect_to_clicked(self.reset_transformation)

        button_copy = CustomPushButton("Copy to clipboard", 90)
        button_copy.connect_to_clicked(self.copy_to_clipboard)

        layout.addWidget(label)
        layout.addWidget(self.matrix_widget)
        layout.addWidget(button_reset)
        layout.addWidget(button_copy)
        layout.addStretch()

        self.ui_repository.signal_transformation_changed.connect(self.update_matrix_display)

    def cell_value_changed(self, row, col, value):
        transformation_matrix = self.ui_repository.transformation_matrix.copy()
        transformation_matrix[row, col] = value
        self.ui_repository.transformation_matrix = transformation_matrix  # Triggers signal

    def set_transformation(self, transformation_matrix):
        self.ui_repository.transformation_matrix = transformation_matrix  # Triggers signal

    def reset_transformation(self):
        self.ui_repository.transformation_matrix = np.eye(4)  # Triggers signal

    def copy_to_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(str(self.ui_repository.transformation_matrix.tolist()))

    def update_matrix_display(self, new_matrix):
        for iRow in range(4):
            for iCol in range(4):
                value = new_matrix[iRow, iCol]
                if float(self.cells[iRow][iCol].text()) == value:
                    continue

                self.cells[iRow][iCol].setText(str(value))
                if not self.cells[iRow][iCol].hasFocus():
                    self.cells[iRow][iCol].setCursorPosition(0)
