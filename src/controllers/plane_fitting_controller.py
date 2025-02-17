from controllers.base_controller import BaseController
from gui.workers.downsampling.qt_plane_fitting import PlaneFittingWorker
from models.data_repository import DataRepository
from params.plane_fitting_params import PlaneFittingParams


class PlaneFittingController(BaseController):
    def __init__(self, repository: DataRepository):
        super().__init__(repository)

    # region Event handlers
    def fit_plane(self, params: PlaneFittingParams):
        pc1 = self.visualizer.o3d_pc1
        pc2 = self.visualizer.o3d_pc2

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

        # TODO: Send signal for the main window to
        for i in range(len(result_data.coefficients_pc1)):
            self.visualizer.add_plane(result_data.coefficients_pc1[i], pc1.get_xyz[result_data.indices_pc1[i]],
                                      result_data.indices_pc1[i], [0.1, 0.8, 0.1], 0)
            self.visualizer.add_plane(result_data.coefficients_pc2[i], pc2.get_xyz[result_data.indices_pc2[i]],
                                      result_data.indices_pc2[i], [0.8, 0.1, 0.1], 1)
    # endregion
