from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QGroupBox, QFormLayout
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from params.io_parameters import LoadRequestParams
from src.gui.widgets.custom_push_button import CustomPushButton
from src.gui.widgets.file_selector_widget import FileSelector


class InputTab(QWidget):
    signal_load_sparse = Signal(LoadRequestParams)
    signal_load_gaussian = Signal(LoadRequestParams)
    signal_load_cached = Signal(LoadRequestParams)

    def __init__(self):
        super().__init__()
        self.progress_dialog = None
        layout_main = QVBoxLayout(self)

        label_io = QLabel("Input and output")
        label_io.setStyleSheet(
            """QLabel {
                font-size: 12pt;
                font-weight: bold;
                padding-bottom: 0.5em;
            }"""
        )

        sparse_group_widget = QGroupBox("Sparse inputs")
        layout_sparse_form = QFormLayout(sparse_group_widget)
        self.fs_input1 = FileSelector()
        self.fs_input2 = FileSelector()
        bt_sparse = CustomPushButton("Import sparse point clouds", 90)
        layout_sparse_form.addRow("First sparse input:", self.fs_input1)
        layout_sparse_form.addRow("Second sparse input:", self.fs_input2)
        layout_sparse_form.addRow(bt_sparse)

        box_widget_cache = QGroupBox("Cached inputs")
        layout_cache = QFormLayout(box_widget_cache)
        self.fs_cache1 = FileSelector()
        self.fs_cache2 = FileSelector()
        bt_cached = CustomPushButton("Import cached point clouds", 90)
        layout_cache.addRow("First cached input:", self.fs_cache1)
        layout_cache.addRow("Second cached input:", self.fs_cache2)
        layout_cache.addRow(bt_cached)

        input_group_widget = QGroupBox("Point cloud inputs")
        layout_input_form = QFormLayout(input_group_widget)
        self.fs_pc1 = FileSelector()
        self.fs_pc2 = FileSelector()
        bt_gaussian = CustomPushButton("Import gaussian point clouds", 90)
        checkbox_cache = QCheckBox()

        layout_input_form.addRow("First point cloud:", self.fs_pc1)
        layout_input_form.addRow("Second point cloud:", self.fs_pc2)
        layout_input_form.addRow("Save converted point clouds:", checkbox_cache)
        layout_input_form.addRow(bt_gaussian)

        layout_main.addWidget(label_io)
        layout_main.addWidget(sparse_group_widget)
        layout_main.addWidget(box_widget_cache)
        layout_main.addWidget(input_group_widget)
        layout_main.addStretch()

        bt_sparse.connect_to_clicked(lambda: self.signal_load_sparse.
                                     emit(LoadRequestParams(self.fs_input1.file_path,
                                                            self.fs_input2.file_path)))
        bt_cached.connect_to_clicked(lambda: self.signal_load_cached
                                     .emit(LoadRequestParams(self.fs_cache1.file_path,
                                                             self.fs_cache2.file_path)))
        bt_gaussian.connect_to_clicked(lambda: self.signal_load_gaussian
                                       .emit(LoadRequestParams(self.fs_pc1.file_path,
                                                               self.fs_pc2.file_path,
                                                               checkbox_cache.isChecked())))
