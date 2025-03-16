import torch

from src.gui.workers.qt_base_worker import BaseWorker
from src.models.gaussian_model import GaussianModel
from src.utils.file_loader import load_plyfile_pc, is_point_cloud_gaussian


class GaussianSaverBase(BaseWorker):
    def __init__(self, transformation, path):
        super().__init__()
        self.transformation = transformation
        self.path = path

    def merge_and_save(self, pc_first, pc_second):
        merged = GaussianModel.get_merged_gaussian_point_clouds(pc_first, pc_second,
                                                                self.transformation)
        torch.cuda.empty_cache()
        merged.save_ply(self.path)
        del merged
        torch.cuda.empty_cache()
        self.signal_progress.emit(100)
        self.signal_finished.emit()


class GaussianSaverNormal(GaussianSaverBase):

    def __init__(self, pc_first, pc_second, transformation, path):
        super().__init__(transformation, path)
        self.pc_first = pc_first
        self.pc_second = pc_second

    def run(self):
        self.merge_and_save(self.pc_first, self.pc_second)


class GaussianSaverUseCorresponding(GaussianSaverBase):
    def __init__(self, pc_path_first, pc_path_second, transformation, path):
        super().__init__(transformation, path)
        self.pc_path_first = pc_path_first
        self.pc_path_second = pc_path_second

    def run(self):
        pc_first_ply = load_plyfile_pc(self.pc_path_first)
        pc_second_ply = load_plyfile_pc(self.pc_path_second)
        if (not pc_first_ply or not pc_second_ply or
                not is_point_cloud_gaussian(pc_first_ply) or not is_point_cloud_gaussian(pc_second_ply)):
            self.signal_error.emit(
                ["Importing one or both of the point clouds failed.\nPlease check that you entered the "
                 "correct path and the point clouds selected are Gaussian point clouds!"])
            self.signal_progress.emit(100)
            self.signal_finished.emit()
            return

        pc_first = GaussianModel(3)
        pc_second = GaussianModel(3)
        pc_first.from_ply(pc_first_ply)
        pc_second.from_ply(pc_second_ply)

        self.merge_and_save(pc_first, pc_second)
