from PySide6.QtCore import Signal

from controllers.base_controller import BaseController
from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.io.qt_gaussian_saver import GaussianSaverUseCorresponding, GaussianSaverNormal
from gui.workers.io.qt_pc_loaders import PointCloudLoaderInput, PointCloudLoaderGaussian, PointCloudLoaderO3D, \
    PointCloudSaver
from gui.workers.qt_base_worker import move_worker_to_thread
from models.data_repository import DataRepository
from params.io_parameters import PointCloudState, LoadRequestParams, SaveRequestParams


class PointCloudIOController(BaseController):
    load_point_clouds_signal = Signal(PointCloudState)

    def __init__(self, repository: DataRepository):
        super().__init__(repository)

    # region Event handlers
    def handle_sparse_load(self, params: LoadRequestParams):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Loading point clouds...")
        worker = PointCloudLoaderInput(params.first_path, params.second_path)
        thread = move_worker_to_thread(self, worker, self.handle_result_sparse,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def handle_gaussian_load(self, params: LoadRequestParams):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Loading point clouds...")
        worker = PointCloudLoaderGaussian(params.first_path, params.second_path)
        thread = move_worker_to_thread(self, worker,
                                       lambda result: self.handle_result_gaussian(result, params.save_converted),
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def handle_cached_load(self, params: LoadRequestParams):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Loading point clouds...")
        worker = PointCloudLoaderO3D(params.first_path, params.second_path)
        thread = move_worker_to_thread(self, worker, self.handle_result_cached,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def merge_point_clouds(self, params: SaveRequestParams):
        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Saving merged point cloud...")

        worker = self.__get_merge_worker(params)
        if worker is None:
            return

        thread = move_worker_to_thread(self, worker, lambda *args: None, error_handler=self.throw_single_error,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    # endregion

    # region Result handlers
    def handle_result_base(self, pc_first, pc_second, original1=None, original2=None):
        error_message = ('Importing one or both of the point clouds failed.\nPlease check that you entered the correct '
                         'path and the point clouds are of the appropriate type!')
        if not pc_first or not pc_second:
            self.throw_single_error(error_message)

        self.repository.current_index = 0

        self.repository.pc_gaussian_list_first.clear()
        self.repository.pc_gaussian_list_second.clear()
        self.repository.pc_open3d_list_first.clear()
        self.repository.pc_open3d_list_second.clear()

        self.repository.pc_gaussian_list_first.append(original1)
        self.repository.pc_gaussian_list_second.append(original2)
        self.repository.pc_open3d_list_first.append(pc_first)
        self.repository.pc_open3d_list_second.append(pc_second)

        pc_loading_params = PointCloudState(pc_first, pc_second, original1, original2, True)
        self.load_point_clouds_signal.emit(pc_loading_params)

        self.update_ui()

    def handle_result_sparse(self, sparse_result: PointCloudLoaderInput.ResultData):
        self.handle_result_base(sparse_result.point_cloud_first, sparse_result.point_cloud_second)

    def handle_result_gaussian(self, gaussian_result: PointCloudLoaderGaussian.ResultData, save_o3d_point_clouds):
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

    def handle_result_cached(self, cached_result: PointCloudLoaderO3D.ResultData):
        self.handle_result_base(cached_result.point_cloud_first, cached_result.point_cloud_first)

    # endregion

    def __get_merge_worker(self, params: SaveRequestParams):
        if params.use_corresponding_pc:
            return GaussianSaverUseCorresponding(params.first_path, params.second_path,
                                                 params.transformation_matrix, params.save_path)

        if self.repository.pc_gaussian_list_first and self.repository.pc_gaussian_list_second:
            index = self.repository.current_index
            pc_first, pc_second = (self.repository.pc_gaussian_list_first[index],
                                   self.repository.pc_gaussian_list_second[index])
            return GaussianSaverNormal(pc_first, pc_second, params.transformation_matrix, params.save_path)

        self.throw_single_error(
            "There were no preloaded point clouds found! Load a Gaussian point cloud before merging, "
            "or check the \"corresponding inputs\" option and select the point clouds you wish to merge."
        )
        return None
