from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QLabel, QVBoxLayout, QComboBox, QWidget, QCheckBox, QSizePolicy, \
    QScrollArea, QStackedWidget, QFormLayout, QGroupBox, QFrame
from open3d.cpu.pybind.pipelines.registration import CorrespondenceCheckerBasedOnEdgeLength, \
    CorrespondenceCheckerBasedOnDistance, CorrespondenceCheckerBasedOnNormal

from params.registration_parameters import FGRRegistrationParams, RANSACRegistrationParams
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.optional_value_widget import OptionalInputField
from src.gui.widgets.simple_input_field_widget import SimpleInputField
from src.utils.global_registration_util import GlobalRegistrationType, RANSACEstimationMethod


class GlobalRegistrationTab(QWidget):
    signal_do_ransac = Signal(RANSACRegistrationParams)
    signal_do_fgr = Signal(FGRRegistrationParams)

    def __init__(self):
        super().__init__()

        # Inputs for Fast Global Registration arguments
        self.checkbox_tuple_test = None
        self.max_tuple_count_widget = None
        self.tuple_scale_widget = None
        self.max_iterations_fgr_widget = None
        self.maximum_correspondence_fgr_widget = None
        self.checkbox_decrease_mu = None
        self.checkbox_use_absolute_scale = None
        self.division_factor_widget = None

        # Inputs for RANSAC arguments
        self.max_iterations_ransac_widget = None
        self.confidence_widget = None
        self.normal_checker = None
        self.distance_checker = None
        self.edge_length_checker = None
        self.ransac_iteration_widget = None
        self.combobox_estimation_method = None
        self.max_correspondence_ransac_widget = None
        self.checkbox_mutual = None

        self.double_validator = QDoubleValidator(0.0, 9999.0, 10)
        self.int_validator = QIntValidator(0, 999999999)

        registration_layout = QVBoxLayout(self)

        scroll_widget = QScrollArea()
        scroll_widget.setFrameShadow(QFrame.Shadow.Plain)
        scroll_widget.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget.setWidgetResizable(True)

        inner_widget = QWidget()
        layout = QVBoxLayout(inner_widget)
        scroll_widget.setWidget(inner_widget)

        # Stack for switching between RANSAC and FGR
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        widget_global = QGroupBox()
        layout_global = QFormLayout(widget_global)

        self.combo_box_global = QComboBox()
        self.combo_box_global.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.combo_box_global.currentIndexChanged.connect(self.global_type_changed)

        for enum_member in GlobalRegistrationType:
            self.combo_box_global.addItem(enum_member.instance_name)

        # Voxel size for downsampling
        self.voxel_size_widget = SimpleInputField("0.05", validator=self.double_validator)

        layout_global.addRow("Global registration type:", self.combo_box_global)
        layout_global.addRow("Voxel size for downsampling:", self.voxel_size_widget)

        # Stack for RANSAC
        self.ransac_widget = self.create_ransac_stack_widget()
        self.ransac_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Stack for FGR
        self.fgr_widget = self.create_fgr_stack_widget()
        self.fgr_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.stack.addWidget(self.ransac_widget)
        self.stack.addWidget(self.fgr_widget)
        self.stack.setCurrentIndex(0)

        bt_apply = CustomPushButton("Start global registration", 100)
        bt_apply.connect_to_clicked(self.registration_button_pressed)

        label_title = QLabel("Global registration")
        label_title.setStyleSheet(
            "QLabel {"
            "    font-size: 12pt;"
            "    font-weight: bold;"
            f"    padding-bottom: 0.5em;"
            "}"
        )

        layout.addWidget(label_title)
        layout.addWidget(widget_global)
        layout.addWidget(self.stack, stretch=1)
        layout.addStretch()

        registration_layout.addWidget(scroll_widget)
        registration_layout.addWidget(bt_apply)

    def create_ransac_stack_widget(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        options_widget = QGroupBox("Options")
        layout_options = QFormLayout(options_widget)

        # Create RANSAC params class with default parameters
        params = RANSACRegistrationParams()

        self.checkbox_mutual = QCheckBox()
        self.checkbox_mutual.setChecked(params.mutual_filter)

        self.max_correspondence_ransac_widget = SimpleInputField(str(params.max_correspondence), validator=self.double_validator)

        self.combobox_estimation_method = QComboBox()
        self.combobox_estimation_method.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        for enum_member in RANSACEstimationMethod:
            self.combobox_estimation_method.addItem(enum_member.instance_name)
        self.combobox_estimation_method.setCurrentIndex(params.estimation_method.value)

        self.ransac_iteration_widget = SimpleInputField(str(params.ransac_n), validator=self.int_validator)

        layout_options.addRow("Estimation type:", self.combobox_estimation_method)
        layout_options.addRow("Maximum correspondence:", self.max_correspondence_ransac_widget)
        layout_options.addRow("RANSAC iterations:", self.ransac_iteration_widget)
        layout_options.addRow("Mutual filtering", self.checkbox_mutual)

        # Checkers
        self.edge_length_checker = OptionalInputField("Edge Length:", "0.0", 75, validator=self.double_validator)
        self.distance_checker = OptionalInputField("Distance:", "0.0", 75, validator=self.double_validator)
        self.normal_checker = OptionalInputField("Normal:", "0.0", 75, validator=self.double_validator)

        checker_widget = QGroupBox("Alignment checker")
        checker_layout = QVBoxLayout(checker_widget)
        checker_layout.addWidget(self.edge_length_checker)
        checker_layout.addWidget(self.distance_checker)
        checker_layout.addWidget(self.normal_checker)

        # Convergence criteria
        convergence_widget = QGroupBox("Convergence criteria")
        layout_convergence = QFormLayout(convergence_widget)
        self.confidence_widget = SimpleInputField(str(params.confidence), validator=self.double_validator)
        self.max_iterations_ransac_widget = SimpleInputField(str(params.max_iteration), validator=self.int_validator)
        layout_convergence.addRow("Confidence:", self.confidence_widget)
        layout_convergence.addRow("Max iterations:", self.max_iterations_ransac_widget)

        layout.addWidget(options_widget)
        layout.addWidget(checker_widget)
        layout.addWidget(convergence_widget)

        return widget

    def create_fgr_stack_widget(self):
        widget_options = QGroupBox("Options")
        widget_options.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout_options = QFormLayout(widget_options)

        # Create FGR params with default arguments
        params = FGRRegistrationParams()

        self.division_factor_widget = SimpleInputField(str(params.division_factor), validator=self.double_validator)
        self.checkbox_use_absolute_scale = QCheckBox()
        self.checkbox_use_absolute_scale.setChecked(params.use_absolute_scale)
        self.checkbox_decrease_mu = QCheckBox()
        self.checkbox_decrease_mu.setChecked(params.decrease_mu)
        self.maximum_correspondence_fgr_widget = SimpleInputField(str(params.maximum_correspondence),
                                                                  validator=self.double_validator)
        self.max_iterations_fgr_widget = SimpleInputField(str(params.max_iterations), validator=self.int_validator)
        self.tuple_scale_widget = SimpleInputField(str(params.tuple_scale), validator=self.double_validator)
        self.max_tuple_count_widget = SimpleInputField(str(params.max_tuple_count), validator=self.int_validator)

        self.checkbox_tuple_test = QCheckBox()
        self.checkbox_tuple_test.setChecked(params.tuple_test)

        layout_options.addRow("Division factor:", self.division_factor_widget)
        layout_options.addRow("Use absolute scale:", self.checkbox_use_absolute_scale)
        layout_options.addRow("Decrease mu:", self.checkbox_decrease_mu)
        layout_options.addRow("Maximum correspondence:", self.maximum_correspondence_fgr_widget)
        layout_options.addRow("Iteration number:", self.max_iterations_fgr_widget)
        layout_options.addRow("Tuple scale:", self.tuple_scale_widget)
        layout_options.addRow("Max tuple count:", self.max_tuple_count_widget)
        layout_options.addRow("Tuple test:", self.checkbox_tuple_test)

        return widget_options

    def global_type_changed(self, index):
        self.stack.setCurrentIndex(index)
        current_widget = self.stack.currentWidget()
        if current_widget is not None:
            height = current_widget.sizeHint().height()
            self.stack.setFixedHeight(height)

    def registration_button_pressed(self):
        if self.combo_box_global.currentIndex() == 0:
            self.emit_ransac_signal()
            return

        self.emit_fgr_signal()

    def emit_ransac_signal(self):
        voxel_size = float(self.voxel_size_widget.lineedit.text())
        mutual_filter = self.checkbox_mutual.isChecked()
        max_correspondence = float(self.max_correspondence_ransac_widget.lineedit.text())
        estimation_method = RANSACEstimationMethod(self.combobox_estimation_method.currentIndex())
        ransac_n = int(self.ransac_iteration_widget.lineedit.text())
        checkers = self.get_ransac_checkers_list()
        max_iteration = int(self.max_iterations_ransac_widget.lineedit.text())
        confidence = float(self.confidence_widget.lineedit.text())
        params = RANSACRegistrationParams(voxel_size, mutual_filter, max_correspondence, estimation_method,
                                          ransac_n, checkers, max_iteration, confidence)
        self.signal_do_ransac.emit(params)

    def emit_fgr_signal(self):
        voxel_size = float(self.voxel_size_widget.lineedit.text())
        division_factor = float(self.division_factor_widget.lineedit.text())
        use_absolute_scale = self.checkbox_use_absolute_scale.isChecked()
        decrease_mu = self.checkbox_decrease_mu.isChecked()
        maximum_correspondence = float(self.maximum_correspondence_fgr_widget.lineedit.text())
        max_iterations = int(self.max_iterations_fgr_widget.lineedit.text())
        tuple_scale = float(self.tuple_scale_widget.lineedit.text())
        max_tuple_count = int(self.max_tuple_count_widget.lineedit.text())
        tuple_test = self.checkbox_tuple_test.isChecked()
        params = FGRRegistrationParams(voxel_size, division_factor, use_absolute_scale, decrease_mu,
                                       maximum_correspondence,
                                       max_iterations, tuple_scale, max_tuple_count, tuple_test)
        self.signal_do_fgr.emit(params)

    def get_ransac_checkers_list(self):
        checkers = []
        if self.distance_checker.is_checked():
            checkers.append(CorrespondenceCheckerBasedOnDistance(float(self.distance_checker.get_value())))
        if self.edge_length_checker.is_checked():
            checkers.append(CorrespondenceCheckerBasedOnEdgeLength(float(self.edge_length_checker.get_value())))
        if self.normal_checker.is_checked():
            checkers.append(CorrespondenceCheckerBasedOnNormal(float(self.normal_checker.get_value())))
        return checkers
