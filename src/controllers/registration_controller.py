import math

import numpy as np
from PySide6.QtWidgets import QMessageBox

from controllers.base_controller import BaseController
from gui.widgets.progress_dialog_factory import ProgressDialogFactory
from gui.workers.graphics.qt_evaluator import RegistrationEvaluator
from gui.workers.qt_base_worker import move_worker_to_thread
from gui.workers.registration.qt_fgr_registrator import FGRRegistrator
from gui.workers.registration.qt_local_registrator import LocalRegistrator
from gui.workers.registration.qt_multiscale_registrator import MultiScaleRegistratorMixture, MultiScaleRegistratorVoxel
from gui.workers.registration.qt_ransac_registrator import RANSACRegistrator
from models.data_repository import DataRepository
from models.ui_state_repository import UIStateRepository


class RegistrationController(BaseController):

    def __init__(self, data_repository: DataRepository, ui_repository: UIStateRepository):
        super().__init__(data_repository, ui_repository)

    # region Event handlers
    def execute_local_registration(self, registration_type, max_correspondence,
                                   relative_fitness, relative_rmse, max_iteration, rejection_type, k_value):
        pc1 = self.data_repository.pc_open3d_list_first[0]
        pc2 = self.data_repository.pc_open3d_list_second[0]
        init_trans = self.ui_repository.transformation_matrix

        # Create worker for local registration
        worker = LocalRegistrator(pc1, pc2, init_trans, registration_type, max_correspondence,
                                  relative_fitness, relative_rmse, max_iteration, rejection_type,
                                  k_value)

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Registering point clouds...")
        thread = move_worker_to_thread(self, worker, self.handle_registration_result_local,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def execute_ransac_registration(self, voxel_size, mutual_filter, max_correspondence, estimation_method,
                                    ransac_n, checkers, max_iteration, confidence):
        pc1 = self.data_repository.pc_open3d_list_first[0]
        pc2 = self.data_repository.pc_open3d_list_second[0]

        worker = RANSACRegistrator(pc1, pc2, self.ui_repository.transformation_matrix,
                                   voxel_size, mutual_filter, max_correspondence,
                                   estimation_method, ransac_n, checkers, max_iteration, confidence)

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Registering point clouds...")
        thread = move_worker_to_thread(self, worker, self.handle_registration_result_global,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def execute_fgr_registration(self, voxel_size, division_factor, use_absolute_scale, decrease_mu,
                                 maximum_correspondence,
                                 max_iterations, tuple_scale, max_tuple_count, tuple_test):
        pc1 = self.data_repository.pc_open3d_list_first[0]
        pc2 = self.data_repository.pc_open3d_list_second[0]

        worker = FGRRegistrator(pc1, pc2, self.ui_repository.transformation_matrix,
                                voxel_size, division_factor, use_absolute_scale, decrease_mu,
                                maximum_correspondence,
                                max_iterations, tuple_scale, max_tuple_count, tuple_test)

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Registering point clouds...")
        thread = move_worker_to_thread(self, worker, self.handle_registration_result_global,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def execute_multiscale_registration(self, use_corresponding, sparse_first, sparse_second, registration_type,
                                        relative_fitness, relative_rmse, voxel_values, iter_values, rejection_type,
                                        k_value, use_mixture):

        if use_mixture:
            pc1_list = self.data_repository.pc_open3d_list_first
            pc2_list = self.data_repository.pc_open3d_list_second
            worker = MultiScaleRegistratorMixture(pc1_list, pc2_list,
                                                  self.ui_repository.transformation_matrix,
                                                  use_corresponding, sparse_first, sparse_second,
                                                  registration_type, relative_fitness,
                                                  relative_rmse, voxel_values, iter_values,
                                                  rejection_type, k_value)
        else:
            pc1 = self.data_repository.pc_open3d_list_first[0]
            pc2 = self.data_repository.pc_open3d_list_second[1]
            worker = MultiScaleRegistratorVoxel(pc1, pc2, self.ui_repository.transformation_matrix,
                                                use_corresponding, sparse_first, sparse_second,
                                                registration_type, relative_fitness,
                                                relative_rmse, voxel_values, iter_values,
                                                rejection_type, k_value)

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Registering point clouds...")
        thread = move_worker_to_thread(self, worker, self.handle_registration_result_local,
                                       self.signal_single_error.emit,
                                       progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def evaluate_registration(self, camera_list, image_path, log_path, color, use_gpu):
        pc1 = self.data_repository.pc_gaussian_list_first[self.data_repository.current_index]
        pc2 = self.data_repository.pc_gaussian_list_second[self.data_repository.current_index]

        if not pc1 or not pc2:
            self.signal_single_error.emit("There are no gaussian point clouds loaded for registration evaluation!"
                                          "\nPlease load two point clouds for registration and evaluation")
            return

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Evaluating registration...")
        worker = RegistrationEvaluator(pc1, pc2, self.ui_repository.transformation_matrix,
                                       camera_list, image_path, log_path, color,
                                       self.data_repository.local_registration_data,
                                       use_gpu)
        progress_dialog.canceled.connect(worker.cancel_evaluation)
        thread = move_worker_to_thread(self, worker, self.handle_evaluation_result,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    # endregion

    # region Result handlers
    def handle_registration_result_local(self, resultData: LocalRegistrator.ResultData):
        self.data_repository.local_registration_data = resultData.registration_data
        results = resultData.result
        self.handle_registration_result_base(results.transformation, results.fitness, results.inlier_rmse)

    def handle_registration_result_global(self, results):
        transformation_actual = np.dot(results.transformation, self.ui_repository.transformation_matrix)
        self.handle_registration_result_base(transformation_actual, results.fitness, results.inlier_rmse)

    def handle_registration_result_base(self, transformation, fitness, inlier_rmse):
        self.ui_repository.transformation_matrix = transformation

        title = "Successful registration"
        message = f"The registration of the point clouds is finished.\n"\
                  f'The transformation will be applied.\n\n'\
                  f"Fitness: {fitness}\n"\
                  f"RMSE: {inlier_rmse}\n"

        self.signal_success_message.emit(title, message, "")

    def handle_evaluation_result(self, log_object):
        title = "Evaluation finished"
        message = "The evaluation finished with"
        if not math.isnan(log_object.psnr):
            message += " success.\n"
            message += f"\nMSE:  {log_object.mse}"
            message += f"\nRMSE: {log_object.rmse}"
            message += f"\nSSIM: {log_object.ssim}"
            message += f"\nPSNR: {log_object.psnr}"
            message += f"\nLPIP: {log_object.lpips}"
        else:
            message += " error."

        detailed_text = []
        if log_object.error_list:
            message += "\nClick \"Show details\" for any potential issues."
            detailed_text.append("\n".join(log_object.error_list))

        self.signal_success_message.emit(title, message, detailed_text)
    # endregion
