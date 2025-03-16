import copy

import numpy as np
import torch

from controllers.base_controller import BaseController
from gui.workers.downsampling.qt_plane_fitting import PlaneFittingWorker
from models.data_repository import DataRepository
from models.ui_state_repository import UIStateRepository
from params.plane_fitting_params import PlaneFittingParams
from utils.plane_fitting_util import get_o3d_plane


class PlaneFittingController(BaseController):
    def __init__(self, data_repository: DataRepository, ui_repository: UIStateRepository):
        super().__init__(data_repository, ui_repository)

    # region Event handlers
    def fit_plane(self, params: PlaneFittingParams):
        pc1 = copy.deepcopy(self.data_repository.pc_open3d_list_first[0])
        pc2 = self.data_repository.pc_open3d_list_second[0]
        pc1.transform(self.ui_repository.transformation_matrix)

        worker = PlaneFittingWorker(pc1, pc2, params.plane_count, params.iteration,
                                    params.distance_threshold, params.normal_threshold, params.min_distance)

        BaseController.run_worker(self, worker, self.handle_fit_plane_result,
                                  "Loading", "Plane fitting in progress...")

    def clear_planes(self):
        self.data_repository.first_plane_indices.clear()
        self.data_repository.second_plane_indices.clear()
        self.data_repository.first_plane_coefficients.clear()
        self.data_repository.second_plane_coefficients.clear()
        self.data_repository.planes = []  # We need to do it this way to trigger the signal in the setter

    # endregion

    # region Result handlers
    def handle_fit_plane_result(self, result_data: PlaneFittingWorker.ResultData):
        pc1 = copy.deepcopy(self.data_repository.pc_open3d_list_first[0])
        pc2 = self.data_repository.pc_open3d_list_second[0]
        pc1.transform(self.ui_repository.transformation_matrix)

        points_tensor1 = torch.from_numpy(np.array(pc1.points, dtype=np.float32))
        points_tensor2 = torch.from_numpy(np.array(pc2.points, dtype=np.float32))

        planes = []
        for i in range(len(result_data.coefficients_pc1)):
            planes.append(get_o3d_plane(result_data.coefficients_pc1[i], points_tensor1[result_data.indices_pc1[i]],
                                        [0.1, 0.8, 0.1]))
            planes.append(get_o3d_plane(result_data.coefficients_pc2[i], points_tensor2[result_data.indices_pc2[i]],
                                        [0.8, 0.1, 0.1]))

        self.data_repository.first_plane_indices = result_data.indices_pc1
        self.data_repository.second_plane_indices = result_data.indices_pc2
        self.data_repository.first_plane_coefficients = result_data.coefficients_pc1
        self.data_repository.second_plane_coefficients = result_data.coefficients_pc2
        self.data_repository.planes = planes
    # endregion
