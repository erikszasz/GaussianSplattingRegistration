import copy

import open3d as o3d
from PySide6 import QtWidgets

from src.gui.workers.qt_base_worker import BaseWorker
from src.models.registration_data import MultiScaleRegistrationData
from src.utils.file_loader import load_sparse_pc
from src.utils.local_registration_util import do_icp_registration


class MultiScaleRegistratorBase(BaseWorker):
    class ResultData:
        def __init__(self, result, registration_data: MultiScaleRegistrationData):
            self.result = result
            self.registration_data = registration_data

    def __init__(self, init_trans, use_corresponding, sparse_first, sparse_second, registration_type, relative_fitness,
                 relative_rmse, voxel_values, iter_values, rejection_type, k_value):
        super().__init__()
        self.init_trans = init_trans
        self.use_corresponding = use_corresponding
        self.sparse_first_path = sparse_first
        self.sparse_second_path = sparse_second
        self.registration_type = registration_type
        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.voxel_values = voxel_values
        self.iter_values = iter_values
        self.rejection_type = rejection_type
        self.k_value = k_value

        self.current_progress = 0
        self.max_progress = len(iter_values)
        self.max_progress += 1 if self.use_corresponding else 0
        self.signal_cancel = False

    def run(self):
        current_trans = self.init_trans

        if self._check_valid_data() is False:
            self.emit_finished()
            return

        if self.use_corresponding:
            current_trans = self.__register_sparse_point_clouds()

        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.emit_finished()
            return

        results = self._register_main_point_clouds(current_trans)
        if results is None:
            self.emit_finished()
            return

        registration_data = self._create_dataclass_object(results)
        self.signal_result.emit(MultiScaleRegistratorBase.ResultData(results, registration_data))
        self.signal_finished.emit()

    def update_progress(self):
        self.current_progress += 1
        new_percent = int(self.current_progress / self.max_progress * 100)
        self.signal_progress.emit(new_percent)

    def cancel(self):
        self.signal_cancel = True

    def emit_finished(self):
        self.signal_finished.emit()
        self.signal_progress.emit(100)

    def __register_sparse_point_clouds(self):
        sparse_pc1 = load_sparse_pc(self.sparse_first_path)
        sparse_pc2 = load_sparse_pc(self.sparse_second_path)

        if not sparse_pc1 or not sparse_pc2:
            self.signal_error.emit(["Point clouds provided as sparse were of a different type"])
            self.emit_finished()
            return None

        # Use first correspondence and iterations from list
        sparse_result = do_icp_registration(sparse_pc1, sparse_pc2, self.init_trans, self.registration_type,
                                            self.voxel_values[0], self.relative_fitness,
                                            self.relative_rmse, self.iter_values[0],
                                            self.rejection_type, self.k_value)

        self.update_progress()
        return sparse_result.transformation

    def _check_valid_data(self):
        raise NotImplementedError

    def _register_main_point_clouds(self, initial_transformation):
        raise NotImplementedError

    def _create_dataclass_object(self, results):
        raise NotImplementedError


class MultiScaleRegistratorVoxel(MultiScaleRegistratorBase):

    def __init__(self, pc1, pc2, init_trans, use_corresponding, sparse_first, sparse_second, registration_type,
                 relative_fitness, relative_rmse, voxel_values, iter_values, rejection_type, k_value):
        super().__init__(init_trans, use_corresponding, sparse_first, sparse_second, registration_type,
                         relative_fitness, relative_rmse, voxel_values, iter_values, rejection_type, k_value)
        self.pc1 = copy.deepcopy(pc1)
        self.pc2 = copy.deepcopy(pc2)

    def _check_valid_data(self):
        if len(self.iter_values) != len(self.voxel_values):
            self.signal_error.emit(["The number of iteration and voxel values provided do not match."])
            self.emit_finished()
            return False

        return True

    def _register_main_point_clouds(self, initial_transformation):
        current_trans = initial_transformation
        results = None

        for scale in range(len(self.iter_values)):
            max_iter = self.iter_values[scale]
            radius = self.voxel_values[scale]

            source_down = self.pc1.voxel_down_sample(radius)
            target_down = self.pc2.voxel_down_sample(radius)

            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            try:
                results = do_icp_registration(source_down, target_down, current_trans, self.registration_type,
                                              radius, self.relative_fitness, self.relative_rmse, max_iter,
                                              self.rejection_type, self.k_value)
            except RuntimeError as e:
                error_str = str(e)
                error_str += f"\nSource: \"{str(source_down)}\"\nTarget: \"{str(target_down)}\""
                self.signal_error.emit([error_str])
                self.emit_finished()
                return None

            self.update_progress()

            current_trans = results.transformation

        return results

    def _create_dataclass_object(self, results):
        return MultiScaleRegistrationData(registration_type=self.registration_type.instance_name,
                                          initial_transformation=self.init_trans,
                                          relative_fitness=self.relative_fitness, relative_rmse=self.relative_rmse,
                                          result_fitness=results.fitness, result_inlier_rmse=results.inlier_rmse,
                                          result_transformation=results.transformation,
                                          voxel_values=self.voxel_values, iteration_values=self.iter_values,
                                          used_sparse_clouds=self.use_corresponding,
                                          used_gaussian_mixtures=False)


class MultiScaleRegistratorMixture(MultiScaleRegistratorBase):

    def __init__(self, pc1_list, pc2_list, init_trans, use_corresponding, sparse_first, sparse_second,
                 registration_type,
                 relative_fitness, relative_rmse, voxel_values, iter_values, rejection_type, k_value):
        super().__init__(init_trans, use_corresponding, sparse_first, sparse_second, registration_type,
                         relative_fitness, relative_rmse, voxel_values, iter_values, rejection_type, k_value)
        self.pc1_list = pc1_list
        self.pc2_list = pc2_list

    def _check_valid_data(self):
        if len(self.pc1_list) != len(self.pc2_list):
            self.signal_error.emit(["The two point cloud lists differ in size."])
            self.emit_finished()
            return False

        if len(self.pc1_list) <= 1:
            self.signal_error.emit(["There are no downscaled mixtures. First create Gaussian Mixtures by "
                                    "running the HEM algorithm in the \"Mixture\" tab!"])
            self.emit_finished()
            return False

        if len(self.iter_values) != len(self.voxel_values):
            self.signal_error.emit(["The number of iteration and voxel values provided do not match."])
            self.emit_finished()
            return False

        if len(self.iter_values) != len(self.pc1_list):
            self.signal_error.emit(["The number of iterations and the mixture levels do not match."])
            self.emit_finished()
            return False

        return True

    def _register_main_point_clouds(self, initial_transformation):
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.signal_finished.emit()
            return None

        current_trans = initial_transformation
        results = None
        for index in range(len(self.iter_values)):
            QtWidgets.QApplication.processEvents()
            if self.signal_cancel:
                self.signal_finished.emit()
                return None

            max_iter = self.iter_values[index]
            max_correspondence = self.voxel_values[index]

            pc1 = copy.deepcopy(self.pc1_list[-(index + 1)])
            pc2 = copy.deepcopy(self.pc2_list[-(index + 1)])

            try:
                results = do_icp_registration(pc1, pc2, current_trans, self.registration_type,
                                              max_correspondence, self.relative_fitness, self.relative_rmse, max_iter,
                                              self.rejection_type, self.k_value)
            except RuntimeError as e:
                error_str = str(e)
                error_str += f"\nSource: \"{str(pc1)}\"\nTarget: \"{str(pc2)}\""
                self.signal_error.emit([error_str])
                self.emit_finished()
                return None

            self.update_progress()

            current_trans = results.transformation

        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            return None

        return results

    def _create_dataclass_object(self, results):
        return MultiScaleRegistrationData(registration_type=self.registration_type.instance_name,
                                          initial_transformation=self.init_trans,
                                          relative_fitness=self.relative_fitness, relative_rmse=self.relative_rmse,
                                          result_fitness=results.fitness, result_inlier_rmse=results.inlier_rmse,
                                          result_transformation=results.transformation,
                                          voxel_values=self.voxel_values, iteration_values=self.iter_values,
                                          used_sparse_clouds=self.use_corresponding,
                                          used_gaussian_mixtures=True)
