from PySide6.QtCore import Signal

from controllers.base_controller import BaseController
from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.downsampling.qt_plane_fitting import PlaneFittingWorker
from gui.workers.downsampling.qt_plane_merging import PlaneInlierMergingWorker
from gui.workers.qt_base_worker import move_worker_to_thread
from models.data_repository import DataRepository
from params.plane_fitting_params import PlaneFittingParams
from utils.plane_fitting_util import get_o3d_plane


class PlaneFittingController(BaseController):
    signal_add_planes = Signal(list)

    def __init__(self, repository: DataRepository):
        super().__init__(repository)

    # region Event handlers
    def fit_plane(self, params: PlaneFittingParams):
        # TODO: Actually we should consider the point cloud after the transformation
        pc1 = self.repository.pc_open3d_list_first[0]
        pc2 = self.repository.pc_open3d_list_second[0]

        worker = PlaneFittingWorker(pc1, pc2, params.plane_count, params.iteration,
                                    params.distance_threshold, params.normal_threshold, params.min_distance)

        BaseController.run_worker(self, worker, self.handle_fit_plane_result,
                                  "Loading", "Plane fitting in progress...")

    def clear_planes(self):
        self.repository.first_plane_indices.clear()
        self.repository.second_plane_indices.clear()
        self.repository.first_plane_coefficients.clear()
        self.repository.second_plane_coefficients.clear()
    # endregion

    # region Result handlers
    def handle_fit_plane_result(self, result_data: PlaneFittingWorker.ResultData):
        self.repository.first_plane_indices.clear()
        self.repository.second_plane_indices.clear()
        self.repository.first_plane_coefficients.clear()
        self.repository.second_plane_coefficients.clear()

        self.repository.first_plane_indices.extend(result_data.indices_pc1)
        self.repository.second_plane_indices.extend(result_data.indices_pc2)
        self.repository.first_plane_coefficients.extend(result_data.coefficients_pc1)
        self.repository.second_plane_coefficients.extend(result_data.coefficients_pc2)

        pc1 = self.repository.pc_gaussian_list_first[0]
        pc2 = self.repository.pc_gaussian_list_second[0]

        planes = []
        for i in range(len(result_data.coefficients_pc1)):
            planes.append(get_o3d_plane(result_data.coefficients_pc1[i], pc1.get_xyz[result_data.indices_pc1[i]],
                                        [0.1, 0.8, 0.1]))
            planes.append(get_o3d_plane(result_data.coefficients_pc2[i], pc2.get_xyz[result_data.indices_pc2[i]],
                                        [0.8, 0.1, 0.1]))

        self.signal_add_planes.emit(planes)  # FIXME: Maybe store these in the repository as well? --> Trigger UI update
    # endregion
