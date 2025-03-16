import mixture_bind
from PySide6 import QtWidgets

from src.gui.workers.qt_base_worker import BaseWorker
from src.models.gaussian_mixture_level import GaussianMixtureModel
from src.models.gaussian_model import GaussianModel
from src.utils.point_cloud_converter import convert_gs_to_open3d_pc


class GaussianMixtureWorker(BaseWorker):
    class ResultData:
        def __init__(self, list_gaussian_first, list_gaussian_second,
                     list_open3d_first, list_open3d_second):
            self.list_gaussian_first = list_gaussian_first
            self.list_gaussian_second = list_gaussian_second
            self.list_open3d_first = list_open3d_first
            self.list_open3d_second = list_open3d_second

    def __init__(self, pc1, pc2, hem_reduction, distance_delta, color_delta, decay_rate, cluster_level):
        super().__init__()

        self.hem_reduction = hem_reduction
        self.distance_delta = distance_delta
        self.color_delta = color_delta
        self.decay_rate = decay_rate
        self.cluster_level = cluster_level

        self.gaussian_pc_first = pc1
        self.gaussian_pc_second = pc2

        self.current_progress = 0
        self.max_progress = 6
        self.signal_cancel = False

    def run(self):
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.signal_finished.emit()
            return

        print("Creating Gaussian Mixture Model for the first point cloud.")
        mixture_level_first = mixture_bind.MixtureLevel.CreateMixtureLevel(
            self.gaussian_pc_first.get_xyz.detach().cpu().tolist(),
            self.gaussian_pc_first.get_colors.detach().cpu().tolist(),
            self.gaussian_pc_first.get_raw_opacity.detach().view(-1).cpu().tolist(),
            self.gaussian_pc_first.get_covariance(1).detach().cpu().tolist(),
            self.gaussian_pc_first.get_spherical_harmonics.detach().cpu().tolist())

        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.signal_finished.emit()
            return

        mixture_models_first = mixture_bind.MixtureCreator.CreateMixture(self.cluster_level, self.hem_reduction,
                                                                         self.distance_delta, self.color_delta,
                                                                         self.decay_rate,
                                                                         mixture_level_first)
        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.signal_finished.emit()
            return

        print("Creating Gaussian Mixture Model for the second point cloud.")
        mixture_level_second = mixture_bind.MixtureLevel.CreateMixtureLevel(
            self.gaussian_pc_second.get_xyz.detach().cpu().tolist(),
            self.gaussian_pc_second.get_colors.detach().cpu().tolist(),
            self.gaussian_pc_second.get_raw_opacity.detach().view(-1).cpu().tolist(),
            self.gaussian_pc_second.get_covariance(1).detach().cpu().tolist(),
            self.gaussian_pc_second.get_spherical_harmonics.detach().cpu().tolist())

        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.signal_finished.emit()
            return

        mixture_models_second = mixture_bind.MixtureCreator.CreateMixture(self.cluster_level, self.hem_reduction,
                                                                          self.distance_delta, self.color_delta,
                                                                          self.decay_rate,
                                                                          mixture_level_second)
        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            self.signal_finished.emit()
            return

        list_gaussian_first = []
        list_gaussian_second = []
        list_open3d_first = []
        list_open3d_second = []

        for mixture in mixture_models_first:
            mixture_model = GaussianMixtureModel(*mixture_bind.MixtureLevel.CreatePythonLists(mixture))
            gaussian = GaussianModel(device_name="cuda:0")
            gaussian.from_mixture(mixture_model)
            gaussian.move_to_device("cpu")

            result_open3d = convert_gs_to_open3d_pc(gaussian)
            list_gaussian_first.append(gaussian)
            list_open3d_first.append(result_open3d)

        self.update_progress()

        for mixture in mixture_models_second:
            mixture_model = GaussianMixtureModel(*mixture_bind.MixtureLevel.CreatePythonLists(mixture))
            gaussian = GaussianModel(device_name="cuda:0")
            gaussian.from_mixture(mixture_model)
            gaussian.move_to_device("cpu")

            result_open3d = convert_gs_to_open3d_pc(gaussian)
            list_gaussian_second.append(gaussian)
            list_open3d_second.append(result_open3d)

        self.update_progress()

        self.signal_result.emit(GaussianMixtureWorker.ResultData(list_gaussian_first, list_gaussian_second,
                                list_open3d_first, list_open3d_second))
        self.signal_finished.emit()

    def update_progress(self):
        self.current_progress += 1
        new_percent = int(self.current_progress / self.max_progress * 100)
        self.signal_progress.emit(new_percent)

    def cancel(self):
        self.signal_cancel = True
