from PySide6.QtCore import QObject, Signal

from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.qt_base_worker import move_worker_to_thread
from models.data_repository import DataRepository


class BaseController(QObject):
    signal_single_error = Signal(str)
    signal_list_error = Signal(list)

    signal_ui_update = Signal()

    def __init__(self, repository: DataRepository):
        super().__init__()
        self.repository = repository
        self.visualizer = None

    @staticmethod
    def run_worker(parent, worker, result_callback, progress_title, progress_message):
        progress_dialog = ProgressDialogFactory.get_progress_dialog(progress_title, progress_message)

        thread = move_worker_to_thread(parent, worker, result_callback, progress_handler=progress_dialog.setValue)

        thread.start()
        progress_dialog.exec()

    def throw_single_error(self, error):
        self.signal_single_error.emit(error)

    def throw_list_error(self, error):
        self.signal_list_error.emit(error)

    def update_ui(self):
        self.signal_ui_update.emit()
