from PySide6 import QtCore
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, \
    QComboBox, QScrollArea, QFrame, QFormLayout, QGroupBox, QStackedWidget, QLabel

from params.registration_parameters import LocalRegistrationParams
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.simple_input_field_widget import SimpleInputField
from src.utils.local_registration_util import LocalRegistrationType, KernelLossFunctionType


class LocalRegistrationTab(QWidget):
    signal_do_registration = QtCore.Signal(LocalRegistrationParams)

    def __init__(self):
        super().__init__()
        double_validator = QDoubleValidator(0.0, 9999.0, 10)
        int_validator = QIntValidator(0, 9999)
        registration_layout = QVBoxLayout(self)

        scroll_widget = QScrollArea()
        scroll_widget.setFrameShadow(QFrame.Shadow.Plain)
        scroll_widget.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget.setWidgetResizable(True)

        inner_widget = QWidget()
        layout = QVBoxLayout(inner_widget)
        scroll_widget.setWidget(inner_widget)

        # Create registration params with default values
        params = LocalRegistrationParams()

        self.combo_box_icp = QComboBox()
        self.combo_box_icp.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for enum_member in LocalRegistrationType:
            self.combo_box_icp.addItem(enum_member.instance_name)
        self.combo_box_icp.setCurrentIndex(params.registration_type.value)
        # Max correspondence
        self.correspondence_widget = SimpleInputField(str(params.max_correspondence), 60, double_validator)

        widget_local = QGroupBox()
        layout_local = QFormLayout(widget_local)
        layout_local.addRow("Local registration type:", self.combo_box_icp)
        layout_local.addRow("Max correspondence:", self.correspondence_widget)

        # Relative fitness
        convergence_widget = QGroupBox("Convergence criteria")
        layout_convergence = QFormLayout(convergence_widget)
        self.fitness_widget = SimpleInputField(str(params.relative_fitness), 60, double_validator)  # Relative fitness
        self.rmse_widget = SimpleInputField(str(params.relative_rmse), 60, double_validator)  # Relative RMSE
        self.iteration_widget = SimpleInputField(str(params.max_iteration), 60, int_validator)  # Max iterations
        layout_convergence.addRow("Relative fitness:", self.fitness_widget)
        layout_convergence.addRow("Relative RMSE:", self.rmse_widget)
        layout_convergence.addRow("Max iteration:", self.iteration_widget)

        # Outlier rejection
        outlier_widget = QGroupBox("Robust Kernel outlier rejection")
        outlier_layout = QFormLayout(outlier_widget)

        self.k_value_widget = SimpleInputField(str(params.k_value), 60, validator=double_validator)
        self.k_value_widget.setEnabled(False)

        self.combo_box_outlier = QComboBox()
        self.combo_box_outlier.currentIndexChanged.connect(self.rejection_type_changed)
        self.combo_box_outlier.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for enum_member in KernelLossFunctionType:
            self.combo_box_outlier.addItem(enum_member.instance_name)
        self.combo_box_icp.setCurrentIndex(params.rejection_type.value)

        outlier_layout.addRow("Loss type:", self.combo_box_outlier)
        outlier_layout.addRow("Standard deviation:", self.k_value_widget)

        bt_apply = CustomPushButton("Start local registration", 100)
        bt_apply.connect_to_clicked(self.registration_button_pressed)

        label_title = QLabel("Local Registration")
        label_title.setStyleSheet(
            "QLabel {"
            "    font-size: 12pt;"
            "    font-weight: bold;"
            f"    padding-bottom: 0.5em;"
            "}"
        )

        # Ugly hack, so local and global registration tabs have the same margins...
        layout.addWidget(label_title)
        layout.addWidget(widget_local)
        stack = QStackedWidget()
        widget_options = QWidget()
        layout_options = QVBoxLayout(widget_options)
        layout_options.setContentsMargins(0, 0, 0, 0)
        layout_options.setSpacing(0)
        layout_options.addWidget(convergence_widget)
        layout_options.addWidget(outlier_widget)
        stack.addWidget(widget_options)
        layout.addWidget(stack)
        layout.addStretch()

        registration_layout.addWidget(scroll_widget)
        registration_layout.addWidget(bt_apply)

    def registration_button_pressed(self):
        registration_type = LocalRegistrationType(self.combo_box_icp.currentIndex())
        max_correspondence = float(self.correspondence_widget.lineedit.text())
        relative_fitness = float(self.fitness_widget.lineedit.text())
        relative_rmse = float(self.rmse_widget.lineedit.text())
        max_iteration = int(self.iteration_widget.lineedit.text())
        rejection_type = KernelLossFunctionType(self.combo_box_outlier.currentIndex())
        k_value = float(self.k_value_widget.lineedit.text())
        params = LocalRegistrationParams(registration_type, max_correspondence, relative_fitness, relative_rmse,
                                         max_iteration, rejection_type, k_value)
        self.signal_do_registration.emit(params)

    def rejection_type_changed(self, index):
        self.k_value_widget.setEnabled(index != 0)
