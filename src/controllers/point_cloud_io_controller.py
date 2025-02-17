from PySide6.QtCore import Signal

from controllers.base_controller import BaseController
from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.io.qt_pc_loaders import PointCloudLoaderInput, PointCloudLoaderGaussian, PointCloudLoaderO3D, \
    PointCloudSaver
from gui.workers.qt_base_worker import move_worker_to_thread
from models.data_repository import DataRepository


class PointCloudIOController(BaseController):
    load_point_clouds_signal = Signal(object, object, object, object)

    def __init__(self, repository: DataRepository):
        super().__init__(repository)

    # region Event handlers
    def handle_sparse_load(self, sparse_path_first, sparse_path_second):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Loading point clouds...")
        worker = PointCloudLoaderInput(sparse_path_first, sparse_path_second)
        thread = move_worker_to_thread(self, worker, self.handle_result_sparse,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def handle_gaussian_load(self, gaussian_path_first, gaussian_path_second, save_o3d_pc):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Loading point clouds...")
        worker = PointCloudLoaderGaussian(gaussian_path_first, gaussian_path_second)
        thread = move_worker_to_thread(self, worker, lambda result: self.handle_result_gaussian(result, save_o3d_pc),
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def handle_cached_load(self, cached_path_first, cached_path_second):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Loading point clouds...")
        worker = PointCloudLoaderO3D(cached_path_first, cached_path_second)
        thread = move_worker_to_thread(self, worker, self.handle_result_cached,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    # endregion

    # region Result handlers
    def handle_result_base(self, pc_first, pc_second, original1=None, original2=None):
        error_message = ('Importing one or both of the point clouds failed.\nPlease check that you entered the correct '
                         'path and the point clouds are of the appropriate type!')
        if not pc_first or not pc_second:
            self.handle_error(error_message)

        self.repository.current_index = 0

        self.repository.pc_gaussian_list_first.clear()
        self.repository.pc_gaussian_list_second.clear()
        self.repository.pc_open3d_list_first.clear()
        self.repository.pc_open3d_list_second.clear()

        self.repository.pc_gaussian_list_first.append(original1)
        self.repository.pc_gaussian_list_second.append(original2)
        self.repository.pc_open3d_list_first.append(pc_first)
        self.repository.pc_open3d_list_second.append(pc_second)

        self.load_point_clouds_signal.emit(pc_first, pc_second, original1, original2)

        self.update_ui()

    def handle_result_sparse(self, sparse_result):
        self.handle_result_base(sparse_result.point_cloud_first, sparse_result.point_cloud_second)

    def handle_result_gaussian(self, gaussian_result, save_o3d_point_clouds):
        self.handle_result_base(gaussian_result.o3d_point_cloud_first, gaussian_result.o3d_point_cloud_second,
                                gaussian_result.gaussian_point_cloud_first,
                                gaussian_result.gaussian_point_cloud_second)

        if not save_o3d_point_clouds:
            return

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Saving open3D point clouds...")
        worker = PointCloudSaver(gaussian_result.o3d_point_cloud_first, gaussian_result.o3d_point_cloud_first)
        thread = move_worker_to_thread(self, worker, lambda *args: None, progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def handle_result_cached(self, cached_result):
        self.handle_result_base(cached_result.point_cloud_first, cached_result.point_cloud_first)

    # endregion
