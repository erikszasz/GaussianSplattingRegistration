from PySide6.QtCore import Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGroupBox, QFormLayout

from params.plane_fitting_params import PlaneFittingParams
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.simple_input_field_widget import SimpleInputField


class PlaneFittingTab(QWidget):
    signal_fit_plane = Signal(PlaneFittingParams)
    signal_clear_plane = Signal()
    signal_merge_plane = Signal()

    def __init__(self):
        super().__init__()
        double_validator = QDoubleValidator(0.0, 9999.0, 10)
        int_validator = QIntValidator(0, 100000)
        layout_main = QVBoxLayout(self)

        label_res = QLabel("Plane fitting")
        label_res.setStyleSheet(
            """QLabel {
                font-size: 12pt;
                font-weight: bold;
                padding-bottom: 0.5em;
            }"""
        )

        # Plane fitting options
        self.iterations_widget = SimpleInputField("1000", 60, int_validator)
        self.plane_count = SimpleInputField("3", 60, int_validator)
        self.distance_threshold_widget = SimpleInputField("0.01", 60, double_validator)
        self.normal_threshold_widget = SimpleInputField("0.9", 60, double_validator)
        self.min_distance_widget = SimpleInputField("0.05", 60, double_validator)
        bt_fit_plane = CustomPushButton("Fit plane(s)", 90)
        bt_fit_plane.connect_to_clicked(self.fit_plane_pressed)

        plane_fitting_box = QGroupBox("Plane fitting")
        layout_plane_fitting = QFormLayout(plane_fitting_box)
        layout_plane_fitting.addRow("Plane count:", self.plane_count)
        layout_plane_fitting.addRow("Max iterations:", self.iterations_widget)
        layout_plane_fitting.addRow("Distance threshold:", self.distance_threshold_widget)
        layout_plane_fitting.addRow("Distance threshold:", self.normal_threshold_widget)
        layout_plane_fitting.addRow("Min sample distance:", self.min_distance_widget)
        layout_plane_fitting.addRow(bt_fit_plane)

        bt_merge = CustomPushButton("DEBUG MERGE", 90)
        bt_merge.connect_to_clicked(self.signal_merge_plane.emit)

        bt_clear_plane = CustomPushButton("Clear plane(s)", 90)
        bt_clear_plane.connect_to_clicked(self.signal_clear_plane.emit)

        layout_main.addWidget(label_res)
        layout_main.addWidget(plane_fitting_box)
        layout_main.addWidget(bt_merge)
        layout_main.addWidget(bt_clear_plane)
        layout_main.addStretch()

    def fit_plane_pressed(self):
        plane_count = int(self.plane_count.text())
        iteration = int(self.iterations_widget.text())
        distance_threshold = float(self.distance_threshold_widget.text())
        normal_threshold = float(self.normal_threshold_widget.text())
        min_distance = float(self.min_distance_widget.text())
        self.signal_fit_plane.emit(PlaneFittingParams(plane_count, iteration,
                                                      distance_threshold, normal_threshold, min_distance))
