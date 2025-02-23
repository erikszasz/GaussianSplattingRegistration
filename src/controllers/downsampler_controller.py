from PySide6.QtCore import Signal

from controllers.base_controller import BaseController
from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.downsampling.qt_gaussian_mixture import GaussianMixtureWorker
from gui.workers.downsampling.qt_plane_merging import PlaneInlierMergingWorker
from gui.workers.qt_base_worker import move_worker_to_thread
from models.data_repository import DataRepository
from models.ui_state_repository import UIStateRepository
from params.merge_parameters import GaussianMixtureParams


class DownsamplerController(BaseController):
    signal_update_hem_slider = Signal(int)

    def __init__(self, data_repository: DataRepository, ui_repository: UIStateRepository):
        super().__init__(data_repository, ui_repository)

    # region Event handlers
    def create_mixture(self, params: GaussianMixtureParams):
        pc1 = pc2 = None

        if len(self.data_repository.pc_gaussian_list_first) != 0:
            pc1 = self.data_repository.pc_gaussian_list_first[0]
        if len(self.data_repository.pc_gaussian_list_second) != 0:
            pc2 = self.data_repository.pc_gaussian_list_second[0]

        if not pc1 or not pc2:
            self.signal_single_error.emit("There are no gaussian point clouds loaded! "
                                          "Please load two point clouds to create Gaussian mixtures.")
            return

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Creating Gaussian mixtures...")
        worker = GaussianMixtureWorker(pc1, pc2, params.hem_reduction, params.distance_delta,
                                       params.color_delta, params.decay_rate, params.cluster_level)
        thread = move_worker_to_thread(self, worker, self.handle_mixture_results,
                                       progress_handler=progress_dialog.setValue)
        progress_dialog.canceled.connect(worker.cancel)

        thread.start()
        progress_dialog.exec()

    def merge_plane_inliers(self):
        pc1 = pc2 = None

        if len(self.data_repository.pc_gaussian_list_first) != 0 and len(
                self.data_repository.pc_gaussian_list_second) != 0:
            pc1 = self.data_repository.pc_gaussian_list_first[0]
            pc2 = self.data_repository.pc_gaussian_list_second[0]

        if not pc1 or not pc2:
            self.signal_single_error.emit("There are no gaussian point clouds loaded!"
                                          "Please load two point clouds to create Gaussian mixtures.")
            return

        if len(self.data_repository.planes) == 0:
            self.signal_single_error.emit("There are no fitted planes to merge!"
                                          "Please run plane fitting to merge the inlier points.")
            return

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Plane inlier merging...")
        worker = PlaneInlierMergingWorker(pc1, pc2, self.data_repository.first_plane_indices,
                                          self.data_repository.second_plane_indices)
        thread = move_worker_to_thread(self, worker, self.handle_plane_merge_results,
                                       progress_handler=progress_dialog.setValue)

        thread.start()
        progress_dialog.exec()

    # endregion

    # region Result handlers
    def handle_mixture_results(self, result_data):

        if len(self.data_repository.pc_gaussian_list_first) > 1:
            self.data_repository.pc_gaussian_list_first = self.data_repository.pc_gaussian_list_first[:1]
            self.data_repository.pc_gaussian_list_second = self.data_repository.pc_gaussian_list_second[:1]
            self.data_repository.pc_open3d_list_first = self.data_repository.pc_open3d_list_first[:1]
            self.data_repository.pc_open3d_list_second = self.data_repository.pc_open3d_list_second[:1]

        self.data_repository.pc_open3d_list_first.extend(result_data.list_open3d_first)
        self.data_repository.pc_open3d_list_second.extend(result_data.list_open3d_second)
        self.data_repository.pc_gaussian_list_first.extend(result_data.list_gaussian_first)
        self.data_repository.pc_gaussian_list_second.extend(result_data.list_gaussian_second)

        self.signal_update_hem_slider.emit(len(self.data_repository.pc_open3d_list_first) - 1)

    def handle_plane_merge_results(self, result_data: PlaneInlierMergingWorker.ResultData):
        self.data_repository.first_plane_indices.clear()
        self.data_repository.second_plane_indices.clear()
        self.data_repository.first_plane_coefficients.clear()
        self.data_repository.second_plane_coefficients.clear()
        self.handle_mixture_results(result_data)
        self.ui_repository.transformation_matrix = self.ui_repository.transformation_matrix  # TODO: Fix this
    # endregion
