import copy

from params.registration_parameters import FGRRegistrationParams
from src.gui.workers.qt_base_worker import BaseWorker
from src.utils.global_registration_util import do_fgr_registration


class FGRRegistrator(BaseWorker):

    def __init__(self, pc1, pc2, init_transformation, registration_params: FGRRegistrationParams):
        super().__init__()

        self.pc1 = copy.deepcopy(pc1)
        self.pc2 = copy.deepcopy(pc2)
        self.pc1.transform(init_transformation)
        self.registration_params = registration_params

    def run(self):
        results = do_fgr_registration(self.pc1, self.pc2, self.registration_params)

        self.signal_result.emit(results)
        self.signal_progress.emit(100)
        self.signal_finished.emit()
