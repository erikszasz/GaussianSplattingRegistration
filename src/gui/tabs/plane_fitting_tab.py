from PySide6.QtCore import Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGroupBox, QFormLayout, QDialog, QTabWidget

from gui.tabs.gaussian_mixture_tab import GaussianMixtureTab
from gui.tabs.global_registration_tab import GlobalRegistrationTab
from gui.tabs.local_registration_tab import LocalRegistrationTab
from models.data_repository import DataRepository
from params.merge_parameters import GaussianMixtureParams
from params.plane_fitting_params import PlaneFittingParams
from params.registration_parameters import FGRRegistrationParams, LocalRegistrationParams, RANSACRegistrationParams
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.simple_input_field_widget import SimpleInputField


class PlaneFittingTab(QWidget):
    signal_fit_plane = Signal(PlaneFittingParams)
    signal_clear_plane = Signal()
    signal_merge_plane = Signal(GaussianMixtureParams)

    signal_do_ransac = Signal(RANSACRegistrationParams)
    signal_do_fgr = Signal(FGRRegistrationParams)
    signal_do_registration = Signal(LocalRegistrationParams)

    def __init__(self, data_repository: DataRepository):
        super().__init__()
        self.data_repository = data_repository
        self.popup = None
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
        bt_clear_plane = CustomPushButton("Clear plane(s)", 90)
        bt_clear_plane.connect_to_clicked(self.signal_clear_plane.emit)

        plane_fitting_box = QGroupBox("Plane fitting")
        layout_plane_fitting = QFormLayout(plane_fitting_box)
        layout_plane_fitting.addRow("Plane count:", self.plane_count)
        layout_plane_fitting.addRow("Max iterations:", self.iterations_widget)
        layout_plane_fitting.addRow("Distance threshold:", self.distance_threshold_widget)
        layout_plane_fitting.addRow("Distance threshold:", self.normal_threshold_widget)
        layout_plane_fitting.addRow("Min sample distance:", self.min_distance_widget)
        layout_plane_fitting.addRow(bt_fit_plane)
        layout_plane_fitting.addRow(bt_clear_plane)

        bt_merge = CustomPushButton("Merge inliers", 90)
        bt_merge.connect_to_clicked(self.create_merge_popup)

        bt_register = CustomPushButton("Register inliers", 90)
        bt_register.connect_to_clicked(self.create_register_popup)

        def _update_buttons():
            has_planes = bool(self.data_repository.planes)
            bt_clear_plane.setEnabled(has_planes)
            bt_merge.setEnabled(has_planes)
            bt_register.setEnabled(has_planes)

        _update_buttons()
        self.data_repository.signal_planes_changed.connect(_update_buttons)

        layout_main.addWidget(label_res)
        layout_main.addWidget(plane_fitting_box)
        layout_main.addWidget(bt_merge)
        layout_main.addWidget(bt_register)
        layout_main.addStretch()

    def fit_plane_pressed(self):
        plane_count = int(self.plane_count.text())
        iteration = int(self.iterations_widget.text())
        distance_threshold = float(self.distance_threshold_widget.text())
        normal_threshold = float(self.normal_threshold_widget.text())
        min_distance = float(self.min_distance_widget.text())
        self.signal_fit_plane.emit(PlaneFittingParams(plane_count, iteration,
                                                      distance_threshold, normal_threshold, min_distance))

    def create_register_popup(self):
        self.popup = QDialog(self)
        self.popup.setModal(True)
        self.popup.setWindowTitle("Registration")
        layout = QVBoxLayout(self.popup)
        global_tab = GlobalRegistrationTab()
        local_tab = LocalRegistrationTab()
        tab_widget = QTabWidget()
        tab_widget.addTab(global_tab, "Global")
        tab_widget.addTab(local_tab, "Local")
        global_tab.signal_do_fgr.connect(lambda *args: (self.popup.close(), self.signal_do_fgr.emit(*args)))
        global_tab.signal_do_ransac.connect(lambda *args: (self.popup.close(), self.signal_do_ransac.emit(*args)))
        local_tab.signal_do_registration.connect(lambda *args: (self.popup.close(),
                                                                self.signal_do_registration.emit(*args)))
        layout.addWidget(tab_widget)
        self.popup.exec()

    def create_merge_popup(self):
        self.popup = QDialog(self)
        self.popup.setModal(True)
        self.popup.setWindowTitle("Gaussian Mixture")
        layout = QVBoxLayout(self.popup)
        mixture_tab = GaussianMixtureTab(False)
        mixture_tab.signal_create_mixture.connect(lambda *args: (self.popup.close(), self.signal_merge_plane.emit(*args)))
        layout.addWidget(mixture_tab)
        self.popup.exec()
