from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QErrorMessage, \
    QFileDialog, QGroupBox, QFormLayout, QHBoxLayout

from models.ui_state_repository import UIStateRepository
from params.io_parameters import SaveRequestParams
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.file_selector_widget import FileSelector


class MergeTab(QWidget):
    signal_merge_point_clouds = QtCore.Signal(SaveRequestParams)

    def __init__(self, ui_repository: UIStateRepository):
        super().__init__()
        self.ui_repository = ui_repository

        layout = QVBoxLayout(self)

        label_title = QLabel("Point cloud merging")
        label_title.setStyleSheet(
            """QLabel {
                font-size: 12pt;
                font-weight: bold;
                padding-bottom: 0.5em;
            }"""
        )

        self.group_box_widget = QGroupBox("Use corresponding Gaussians")
        self.group_box_widget.setCheckable(True)
        self.group_box_widget.setChecked(False)
        self.group_box_widget.toggled.connect(self.checkbox_changed)
        layout_group_box = QFormLayout(self.group_box_widget)

        self.fs_input1 = FileSelector()
        self.fs_input2 = FileSelector()
        layout_group_box.addRow("First point cloud:", self.fs_input1)
        layout_group_box.addRow("Second point cloud:", self.fs_input2)

        widget_save = QWidget()
        layout_save = QHBoxLayout(widget_save)
        label_save = QLabel("Save path:")
        self.fs_merge = FileSelector(file_type=QFileDialog.FileMode.AnyFile)
        layout_save.addWidget(label_save)
        layout_save.addWidget(self.fs_merge)
        bt_merge = CustomPushButton("Merge point clouds", 90)
        bt_merge.connect_to_clicked(self.merge_point_clouds)

        layout.addWidget(label_title)
        layout.addWidget(self.group_box_widget)
        layout.addWidget(widget_save)
        layout.addWidget(bt_merge)
        layout.addStretch()

    def checkbox_changed(self, state):
        if state:
            return

        self.fs_input1.inputField.setText("")
        self.fs_input1.file_path = ""

        self.fs_input2.inputField.setText("")
        self.fs_input2.file_path = ""

    def merge_point_clouds(self):
        if not self.fs_merge.file_path:
            dialog = QErrorMessage(self)
            dialog.setModal(True)
            dialog.setWindowTitle("Error")
            dialog.showMessage("Please select location to save the merged point cloud!")
            return

        is_checked = self.group_box_widget.isChecked()
        pc_path1 = self.fs_input1.file_path
        pc_path2 = self.fs_input2.file_path
        merge_path = self.fs_merge.file_path
        transformation = self.ui_repository.transformation_matrix
        self.signal_merge_point_clouds.emit(SaveRequestParams(merge_path, transformation, is_checked, pc_path1, pc_path2))
