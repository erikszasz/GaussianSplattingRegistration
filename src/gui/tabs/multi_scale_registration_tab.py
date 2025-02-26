from PySide6.QtCore import QLocale, QRegularExpression, Signal
from PySide6.QtGui import QDoubleValidator, QRegularExpressionValidator
from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QComboBox, QSizePolicy, QFrame, QScrollArea, QFormLayout,
                               QGroupBox)

from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.file_selector_widget import FileSelector
from src.gui.widgets.simple_input_field_widget import SimpleInputField
from src.utils.local_registration_util import LocalRegistrationType, KernelLossFunctionType


class MultiScaleRegistrationTab(QWidget):
    signal_do_registration = Signal(bool, str, str, LocalRegistrationType, float, float,
                                    list, list,
                                    KernelLossFunctionType, float, bool)

    def __init__(self):
        super().__init__()

        registration_layout = QVBoxLayout()
        self.setLayout(registration_layout)

        scroll_widget = QScrollArea()
        scroll_widget.setFrameShadow(QFrame.Shadow.Plain)
        scroll_widget.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget.setWidgetResizable(True)

        inner_widget = QWidget()
        layout = QVBoxLayout(inner_widget)
        scroll_widget.setWidget(inner_widget)

        # validators
        locale = QLocale(QLocale.Language.C)
        double_validator = QDoubleValidator()
        double_validator.setLocale(locale)
        double_validator.setRange(0.0, 9999.0)
        double_validator.setDecimals(10)

        double_list_validator = QRegularExpressionValidator()
        double_list_validator.setLocale(locale)
        regex_double = QRegularExpression("(?!0\\d)(\\d+(\\.\\d+)?)(,(-?(?!0\\d)(\\d+(\\.\\d+)?)))*")
        double_list_validator.setRegularExpression(regex_double)

        int_list_validator = QRegularExpressionValidator()
        int_list_validator.setLocale(locale)
        regex_int = QRegularExpression("(?!0\\d)\\d+(,(-?(?!0\\d)\\d+))*")
        int_list_validator.setRegularExpression(regex_int)

        label_title = QLabel("Multiscale Local Registration")
        label_title.setStyleSheet(
            "QLabel {"
            "    font-size: 12pt;"
            "    font-weight: bold;"
            f"    padding-bottom: 0.5em;"
            "}"
        )

        self.widget_group_sparse = QGroupBox("Sparse input")
        self.widget_group_sparse.setCheckable(True)
        self.widget_group_sparse.setChecked(False)
        self.widget_group_sparse.toggled.connect(self.checkbox_changed)
        layout_group_sparse = QFormLayout(self.widget_group_sparse)

        # File selectors for sparse input
        self.fs_sparse1 = FileSelector()
        self.fs_sparse2 = FileSelector()

        layout_group_sparse.addRow("First sparse input:", self.fs_sparse1)
        layout_group_sparse.addRow("Second sparse input:", self.fs_sparse2)

        # Local registration type
        widget_convergence_criteria = QGroupBox("Convergence criteria")
        self.layout_convergence_criteria = QFormLayout(widget_convergence_criteria)

        self.combo_box_icp = QComboBox()
        self.combo_box_icp.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for enum_member in LocalRegistrationType:
            self.combo_box_icp.addItem(enum_member.instance_name)

        # Basic input fields
        self.fitness_widget = SimpleInputField("0.000001", 60, double_validator)
        self.rmse_widget = SimpleInputField("0.000001", 60, double_validator)
        self.iter_values = SimpleInputField("50,30,20", 100, int_list_validator)

        self.combo_box_multiscale = QComboBox()
        self.combo_box_multiscale.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.combo_box_multiscale.addItem("Voxel downsampling")
        self.combo_box_multiscale.addItem("HEM Gaussian mixture")
        self.combo_box_multiscale.currentIndexChanged.connect(self.downscale_type_changed)

        # Only, when using voxel downsample
        self.voxel_values = SimpleInputField("5,2.5,2", 100, double_list_validator)

        self.layout_convergence_criteria.addRow("Local registration type:", self.combo_box_icp)
        self.layout_convergence_criteria.addRow("Downscale type:", self.combo_box_multiscale)
        self.layout_convergence_criteria.addRow("Relative fitness:", self.fitness_widget)
        self.layout_convergence_criteria.addRow("Relative RMSE:", self.rmse_widget)
        self.layout_convergence_criteria.addRow("Iteration values:", self.iter_values)
        self.layout_convergence_criteria.addRow("Voxel values:", self.voxel_values)

        # Outlier rejection
        outlier_widget = QGroupBox("Robust Kernel outlier rejection")
        outlier_layout = QFormLayout(outlier_widget)

        self.k_value_widget = SimpleInputField("0.0", 60, validator=double_validator)
        self.k_value_widget.setEnabled(False)

        self.combo_box_outlier = QComboBox(self)
        self.combo_box_outlier.currentIndexChanged.connect(self.rejection_type_changed)
        self.combo_box_outlier.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for enum_member in KernelLossFunctionType:
            self.combo_box_outlier.addItem(enum_member.instance_name)

        outlier_layout.addRow("Loss type:", self.combo_box_outlier)
        outlier_layout.addRow("Standard deviation:", self.k_value_widget)

        bt_apply = CustomPushButton("Start multiscale registration", 100)
        bt_apply.connect_to_clicked(self.registration_button_pressed)

        layout.addWidget(label_title)
        layout.addWidget(self.widget_group_sparse)
        layout.addWidget(widget_convergence_criteria)
        layout.addWidget(outlier_widget)
        layout.addStretch()

        registration_layout.addWidget(scroll_widget)
        registration_layout.addWidget(bt_apply)

    def checkbox_changed(self, state):
        if state:
            return

        self.fs_sparse1.inputField.setText("")
        self.fs_sparse1.file_path = ""

        self.fs_sparse2.inputField.setText("")
        self.fs_sparse2.file_path = ""

    def registration_button_pressed(self):
        use_corresponding = self.widget_group_sparse.isChecked()
        sparse_first = self.fs_sparse1.file_path
        sparse_second = self.fs_sparse2.file_path
        registration_type = LocalRegistrationType(self.combo_box_icp.currentIndex())
        relative_fitness = float(self.fitness_widget.lineedit.text())
        relative_rmse = float(self.rmse_widget.lineedit.text())

        voxel_values = list(map(float, filter(None, self.voxel_values.lineedit.text().split(","))))
        iter_values = list(map(int, filter(None, self.iter_values.lineedit.text().split(","))))

        rejection_type = KernelLossFunctionType(self.combo_box_outlier.currentIndex())
        k_value = float(self.k_value_widget.lineedit.text())
        use_mixture = self.combo_box_multiscale.currentIndex() == 1

        self.signal_do_registration.emit(use_corresponding, sparse_first, sparse_second,
                                         registration_type,
                                         relative_fitness, relative_rmse,
                                         voxel_values, iter_values,
                                         rejection_type, k_value, use_mixture)

    def rejection_type_changed(self, index):
        self.k_value_widget.setEnabled(index != 0)

    def downscale_type_changed(self, index):
        label = self.layout_convergence_criteria.labelForField(self.voxel_values)
        if label is None:
            return

        label.setText("Voxel values:" if index == 0 else "Correspondences:")
        self.voxel_values.update()
