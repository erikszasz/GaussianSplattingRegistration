import json
import os.path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QSpinBox, QLineEdit, QErrorMessage, QCheckBox, QFormLayout)

from src.gui.widgets.color_picker_widget import ColorPicker
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.file_selector_widget import FileSelector
from src.models.camera import Camera
from src.utils.general_utils import convert_to_camera_transform


class EvaluationTab(QWidget):
    signal_camera_change = Signal(np.ndarray)
    signal_evaluate_registration = Signal(list, str, str, np.ndarray, bool)

    def __init__(self):
        super().__init__()
        self.error_message = None
        self.cameras_list = []
        layout = QVBoxLayout(self)

        label_title = QLabel("Evaluation")
        label_title.setStyleSheet(
            """QLabel {
                font-size: 12pt;
                font-weight: bold;
                padding-bottom: 0.5em;
            }"""
        )

        input_group = QGroupBox("Cameras")
        input_layout = QFormLayout(input_group)

        self.fs_cameras = FileSelector(name_filter="*.json")
        self.button_load_cameras = CustomPushButton("Load cameras", 100)

        # Spinbox row
        spinbox_widget = QWidget()
        spinbox_layout = QHBoxLayout(spinbox_widget)
        label = QLabel("Snap to:")

        self.spinbox = QSpinBox()
        self.spinbox.setFixedWidth(70)
        self.spinbox.setFixedHeight(30)
        self.spinbox.setRange(0, 0)
        self.spinbox.setKeyboardTracking(False)
        self.spinbox.setEnabled(False)
        self.spinbox.valueChanged.connect(self.current_camera_changed)

        self.current_image_name = QLineEdit()
        self.current_image_name.setEnabled(False)
        self.current_image_name.setFixedHeight(30)
        self.current_image_name.setFixedWidth(100)
        self.current_image_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        spinbox_layout.addWidget(label)
        spinbox_layout.addWidget(self.spinbox)
        spinbox_layout.addWidget(self.current_image_name)
        spinbox_layout.addStretch()

        input_layout.addRow("Cameras:", self.fs_cameras)
        input_layout.addRow(spinbox_widget)
        input_layout.addRow(self.button_load_cameras)

        evaluation_group = QGroupBox("Evaluation")
        evaluation_layout = QFormLayout(evaluation_group)
        self.fs_images = FileSelector(file_type=QFileDialog.FileMode.Directory)
        self.fs_log = FileSelector(name_filter="Log files (*.txt *.log);;*.txt;;*.log",
                                   file_type=QFileDialog.FileMode.AnyFile)
        self.render_color = ColorPicker(np.zeros(3))
        self.checkbox_gpu = QCheckBox()
        self.button_evaluate = CustomPushButton("Evaluate", 100)

        evaluation_layout.addRow("Images folder:", self.fs_images)
        evaluation_layout.addRow("Log file:", self.fs_log)
        evaluation_layout.addRow("Background color:", self.render_color)
        evaluation_layout.addRow("Use GPU for evaluation:", self.checkbox_gpu)
        evaluation_layout.addRow(self.button_evaluate)

        layout.addWidget(label_title)
        layout.addWidget(input_group)
        layout.addWidget(evaluation_group)
        layout.addStretch()

        self.button_load_cameras.connect_to_clicked(self.load_cameras_clicked)
        self.button_evaluate.connect_to_clicked(self.evaluate_registration)

    def load_cameras_clicked(self):
        self.cameras_list.clear()
        self.spinbox.setEnabled(False)

        cameras_path = self.fs_cameras.file_path
        if cameras_path == "" or not os.path.isfile(cameras_path):
            dialog = QErrorMessage(self)
            dialog.setWindowTitle("Error")
            dialog.setModal(True)
            dialog.showMessage("Select a valid path to your cameras JSON file!")
            return

        f = open(cameras_path)
        data = json.load(f)
        for camera_iter in data:
            fx = camera_iter["fx"]
            fy = camera_iter["fy"]
            height = camera_iter["height"]
            width = camera_iter["width"]

            rot = np.array([np.array(xi) for xi in camera_iter["rotation"]])
            pos = np.array([np.array(xi) for xi in camera_iter["position"]])
            R, T = convert_to_camera_transform(rot, pos)
            image_name = camera_iter["img_name"]
            camera = Camera(R, T, fx, fy, image_name, width, height)
            self.cameras_list.append(camera)

        self.spinbox.setEnabled(True)
        self.spinbox.setRange(1, len(self.cameras_list))
        self.spinbox.setValue(0)
        self.current_image_name.setText(self.cameras_list[0].image_name)

    def current_camera_changed(self, camera_id):
        current_camera = self.cameras_list[camera_id - 1]
        self.current_image_name.setText(current_camera.image_name)
        self.signal_camera_change.emit(current_camera.viewmat.squeeze())

    def evaluate_registration(self):
        image_path = self.fs_images.file_path
        log_file = self.fs_log.file_path

        if not len(self.cameras_list) > 0:
            self.creat_error_box("There are no cameras loaded.\nPlease load cameras before evaluation!")
            return

        if not os.path.isdir(image_path):
            self.creat_error_box("Select a valid image directory!")
            return

        if log_file == "":
            self.creat_error_box("Select a path for the log file!")
            return

        color = np.asarray(self.render_color.color)
        use_gpu = self.checkbox_gpu.isChecked()
        self.signal_evaluate_registration.emit(self.cameras_list, image_path, log_file, color, use_gpu)

    def creat_error_box(self, message):
        self.error_message = QErrorMessage()
        self.error_message.setWindowTitle("Error")
        self.error_message.setModal(True)
        self.error_message.showMessage(message)
