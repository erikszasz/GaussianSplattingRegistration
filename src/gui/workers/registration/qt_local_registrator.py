import copy

from PySide6.QtCore import Signal

from src.gui.workers.qt_base_worker import BaseWorker
from src.models.registration_data import LocalRegistrationData
from src.utils.local_registration_util import do_icp_registration


class LocalRegistrator(BaseWorker):
    signal_registration_done = Signal(object, object)

    class ResultData:
        def __init__(self, result, registration_data: LocalRegistrationData):
            self.result = result
            self.registration_data = registration_data

    def __init__(self, pc1, pc2, init_trans, params):
        super().__init__()

        self.pc1 = copy.deepcopy(pc1)
        self.pc2 = copy.deepcopy(pc2)
        self.init_trans = init_trans
        self.registration_params = params

    def run(self):
        results = do_icp_registration(self.pc1, self.pc2, self.init_trans, self.registration_params)

        dataclass = self.create_dataclass_object(results)
        self.signal_result.emit(LocalRegistrator.ResultData(results, dataclass))
        self.signal_progress.emit(100)
        self.signal_finished.emit()

    def create_dataclass_object(self, results):
        return LocalRegistrationData(registration_type=self.registration_params.registration_type.instance_name,
                                     initial_transformation=self.init_trans,
                                     relative_fitness=self.registration_params.relative_fitness,
                                     relative_rmse=self.registration_params.relative_rmse,
                                     result_fitness=results.fitness, result_inlier_rmse=results.inlier_rmse,
                                     result_transformation=results.transformation,
                                     max_correspondence=self.registration_params.max_correspondence,
                                     max_iteration=self.registration_params.max_iteration)
