from PySide6.QtCore import QObject, Signal

from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.qt_base_worker import move_worker_to_thread
from models.data_repository import DataRepository


class BaseController(QObject):
    error_signal = Signal(str)
    ui_update_signal = Signal()

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

    def handle_error(self, error):
        self.error_signal.emit(error)

    def update_ui(self):
        self.ui_update_signal.emit()
