import numpy as np
from PySide6 import QtCore
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFormLayout, QGroupBox, QPushButton, QSizePolicy, \
    QStyle, QHBoxLayout, QStackedWidget, QErrorMessage

from gui.widgets.animated_toggle_widget import AnimatedToggle
from src.gui.widgets.color_picker_widget import ColorPicker
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.simple_input_field_widget import SimpleInputField
from src.gui.widgets.vector_widget import VectorWidget


class VisualizerTab(QWidget):
    class CameraView:
        def __init__(self, zoom, front, lookat, up):
            self.zoom = zoom
            self.front = front
            self.lookat = lookat
            self.up = up

    signal_change_vis_settings_o3d = QtCore.Signal(CameraView,
                                                   object, object)

    signal_change_vis_settings_3dgs = QtCore.Signal(CameraView, float, float, float, object)

    signal_change_type = QtCore.Signal(int)
    signal_get_current_view = QtCore.Signal()
    signal_pop_visualizer = QtCore.Signal()

    def __init__(self, parent_window):
        super().__init__()
        layout = QVBoxLayout(self)
        double_validator = QDoubleValidator(-9999.0, 9999.0, 10)
        self.parent_window = parent_window

        label_title = QLabel("Visualization")
        label_title.setStyleSheet(
            """QLabel {
                font-size: 12pt;
                font-weight: bold;
                padding-bottom: 0.5em;
            }"""
        )

        checkbox_widget = QWidget()
        checkbox_form = QFormLayout(checkbox_widget)
        checkbox_form.setContentsMargins(0, 0, 0, 0)
        self.checkbox = AnimatedToggle()
        self.checkbox.setFixedSize(self.checkbox.sizeHint())
        self.checkbox.toggled.connect(self.change_visualizer)
        checkbox_text = QLabel("Use GSPLAT:")
        checkbox_text.setStyleSheet("""QLabel {
                font-size: 10pt;
            }""")
        checkbox_form.addRow(checkbox_text, self.checkbox)

        titled_label_widget = QWidget()
        label_layout = QHBoxLayout(titled_label_widget)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(0)

        self.pop_button = QPushButton(self)
        self.pop_button.setCheckable(True)
        self.pop_button.setChecked(False)
        self.pop_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.pop_button.setFixedSize(20, 20)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton)
        self.pop_button.setIcon(icon)

        label_layout.addWidget(label_title)
        label_layout.addWidget(self.pop_button)

        self.pop_button.clicked.connect(self.pop_visualizer)

        self.stack_widget = QStackedWidget()
        self.stack_widget.setContentsMargins(0, 0, 0, 0)
        self.stack_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.form_widget_color = QGroupBox("Debug color")
        self.form_widget_color.setCheckable(True)
        self.form_widget_color.setChecked(False)
        layout_form_color = QFormLayout(self.form_widget_color)

        self.debug_color_dialog_first = ColorPicker()
        self.debug_color_dialog_second = ColorPicker()
        layout_form_color.addRow("Primary debug color:", self.debug_color_dialog_first)
        layout_form_color.addRow("Secondary debug color:", self.debug_color_dialog_second)

        self.stack_widget.addWidget(self.form_widget_color)

        widget_3dgs_settings = QWidget()
        layout_3dgs_settings = QVBoxLayout(widget_3dgs_settings)
        layout_3dgs_settings.setContentsMargins(0, 0, 0, 0)

        groupbox_3dgs_controls = QGroupBox("3DGS settings")
        layout_3dgs_controls = QFormLayout(groupbox_3dgs_controls)

        self.translation_speed_widget = SimpleInputField("7.0", 60, double_validator)
        self.rotation_speed_widget = SimpleInputField("0.01", 60, double_validator)
        self.roll_speed_widget = SimpleInputField("0.1", 60, double_validator)
        self.background_color = ColorPicker(np.asarray([0.09803921568627451, 0.13725490196078433, 0.17647058823529413]))
        layout_3dgs_controls.addRow("Translation speed:", self.translation_speed_widget)
        layout_3dgs_controls.addRow("Rotation speed:", self.rotation_speed_widget)
        layout_3dgs_controls.addRow("Roll speed:", self.roll_speed_widget)
        layout_3dgs_controls.addRow("Background color:", self.background_color)

        self.stack_widget.addWidget(groupbox_3dgs_controls)

        view_group_widget = QGroupBox("View")
        layout_form_view = QFormLayout(view_group_widget)
        self.zoom_widget = SimpleInputField("1.0", 60, validator=double_validator)
        zoom_layout = self.zoom_widget.layout()
        margin = zoom_layout.getContentsMargins()
        zoom_layout.setContentsMargins(margin[0], 0, margin[2], 0)
        self.front_widget = VectorWidget(3, [0.0, 0.0, 1.0], double_validator)
        self.lookat_widget = VectorWidget(3, [0.0, 0.0, 0.0], double_validator)
        self.up_widget = VectorWidget(3, [0.0, 1.0, 0.0], double_validator)
        layout_form_view.addRow("Zoom:", self.zoom_widget)
        layout_form_view.addRow("Front:", self.front_widget)
        layout_form_view.addRow("Look at:", self.lookat_widget)
        layout_form_view.addRow("Up:", self.up_widget)

        button_apply = CustomPushButton("Apply", 90)
        button_apply.connect_to_clicked(self.apply_to_vis)
        button_copy = CustomPushButton("Copy current view", 90)
        button_copy.connect_to_clicked(self.get_current_view)

        self.stack_widget.setFixedHeight(self.form_widget_color.sizeHint().height())

        layout.addWidget(titled_label_widget)
        layout.addWidget(checkbox_widget)
        layout.addWidget(self.stack_widget)
        layout.addWidget(view_group_widget)
        layout.addWidget(button_apply)
        layout.addWidget(button_copy)
        layout.addStretch()

    def apply_to_vis(self):
        if not self.checkbox.isChecked():
            use_debug_color = self.get_use_debug_color()
            dc1 = dc2 = None
            if use_debug_color:
                dc1 = np.asarray(self.debug_color_dialog_first.color)
                dc2 = np.asarray(self.debug_color_dialog_second.color)

            self.signal_change_vis_settings_o3d.emit(self.get_current_transformations(), dc1, dc2)
            return

        roll_speed = float(self.roll_speed_widget.text())
        rotation_speed = float(self.rotation_speed_widget.text())
        translate_speed = float(self.translation_speed_widget.text())
        background_color = np.asarray(self.background_color.color)
        self.signal_change_vis_settings_3dgs.emit(self.get_current_transformations(),
                                                  translate_speed, rotation_speed, roll_speed, background_color)

    def get_current_view(self):
        self.signal_get_current_view.emit()

    def set_visualizer_attributes(self, zoom, front, lookat, up):
        self.zoom_widget.lineedit.setText(str(zoom))
        self.zoom_widget.lineedit.setCursorPosition(0)
        self.front_widget.set_values(front)
        self.lookat_widget.set_values(lookat)
        self.up_widget.set_values(up)

    def get_use_debug_color(self):
        return self.form_widget_color.isChecked()

    def get_debug_colors(self):
        return np.asarray(self.debug_color_dialog_first.color), np.asarray(
            self.debug_color_dialog_second.color)

    def get_current_transformations(self):
        return VisualizerTab.CameraView(float(
            self.zoom_widget.lineedit.text()), self.front_widget.values,
                self.lookat_widget.values, self.up_widget.values)

    def pop_visualizer(self):
        self.signal_pop_visualizer.emit()

    def change_visualizer(self, visualizer_index):
        if visualizer_index and len(self.parent_window.data_repository.pc_open3d_list_first) == 0:
            dialog = QErrorMessage(self)
            dialog.setModal(True)
            dialog.setWindowTitle("Error")
            dialog.showMessage("First load two 3DGS point clouds...")
            self.checkbox.toggle()
            return

        if visualizer_index and self.parent_window.visualizer_window.is_ortho():
            dialog = QErrorMessage(self)
            dialog.setModal(True)
            dialog.setWindowTitle("Error")
            dialog.showMessage("Orthogonal view is not supported!")
            self.checkbox.toggle()
            return

        if self.pop_button.isChecked():
            self.pop_button.setChecked(False)
            self.pop_visualizer()

        self.stack_widget.setCurrentIndex(visualizer_index)
        current_widget = self.stack_widget.currentWidget()
        if current_widget is not None:
            self.stack_widget.setFixedHeight(current_widget.sizeHint().height())

        self.signal_change_type.emit(visualizer_index)
