import json

from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import mixture_bind

from src.models.gaussian_model import GaussianModel
from src.utils.graphics_utils import sh2rgb

from src.models.gaussian_mixture_level import GaussianMixtureModel
from src.utils.point_cloud_converter import convert_gs_to_open3d_pc


class GaussianMixtureWorker(QObject):
    signal_finished = pyqtSignal()
    signal_mixture_created = pyqtSignal(list, list, list, list)

    signal_update_progress = pyqtSignal(int)

    def __init__(self, pc1, pc2, hem_reduction, distance_delta, color_delta, cluster_level):
        super().__init__()

        self.hem_reduction = hem_reduction
        self.distance_delta = distance_delta
        self.color_delta = color_delta
        self.cluster_level = cluster_level

        self.gaussian_pc_first = pc1
        self.gaussian_pc_second = pc2

        self.current_progress = 0
        self.max_progress = 5
        self.signal_cancel = False

    def execute(self):

        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            return

        mixture_level_first = mixture_bind.MixtureLevel.CreateMixtureLevel(
            self.gaussian_pc_first.get_xyz.detach().cpu().tolist(),
            self.gaussian_pc_first.get_colors.detach().cpu().tolist(),
            self.gaussian_pc_first.get_raw_opacity.detach().view(-1).cpu().tolist(),
            self.gaussian_pc_first.get_covariance(1).detach().cpu().tolist(),
            self.gaussian_pc_first.get_spherical_harmonics.detach().cpu().tolist())

        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            return

        mixture_level_second = mixture_bind.MixtureLevel.CreateMixtureLevel(
            self.gaussian_pc_second.get_xyz.detach().cpu().tolist(),
            self.gaussian_pc_second.get_colors.detach().cpu().tolist(),
            self.gaussian_pc_second.get_raw_opacity.detach().view(-1).cpu().tolist(),
            self.gaussian_pc_second.get_covariance(1).detach().cpu().tolist(),
            self.gaussian_pc_second.get_spherical_harmonics.detach().cpu().tolist())

        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            return

        print("Creating Gaussian Mixture Model for the first point cloud.")
        mixture_models_first = mixture_bind.MixtureCreator.CreateMixture(self.cluster_level, self.hem_reduction, self.distance_delta, self.color_delta, mixture_level_first)
        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            return

        print("Creating Gaussian Mixture Model for the second point cloud.")
        mixture_models_second = mixture_bind.MixtureCreator.CreateMixture(self.cluster_level, self.hem_reduction, self.distance_delta, self.color_delta, mixture_level_second)
        self.update_progress()
        QtWidgets.QApplication.processEvents()
        if self.signal_cancel:
            return

        list_gaussian_first = []
        list_gaussian_second = []
        list_open3d_first = []
        list_open3d_second = []

        for mixture in mixture_models_first:
            mixture_model = GaussianMixtureModel(*mixture_bind.MixtureLevel.CreatePythonLists(mixture))
            gaussian = GaussianModel()
            gaussian.from_mixture(mixture_model)

            result_open3d = convert_gs_to_open3d_pc(gaussian)
            list_gaussian_first.append(gaussian)
            list_open3d_first.append(result_open3d)

        self.update_progress()

        for mixture in mixture_models_second:
            mixture_model = GaussianMixtureModel(*mixture_bind.MixtureLevel.CreatePythonLists(mixture))
            gaussian = GaussianModel()
            gaussian.from_mixture(mixture_model)

            result_open3d = convert_gs_to_open3d_pc(gaussian)
            list_gaussian_second.append(gaussian)
            list_open3d_second.append(result_open3d)

        self.update_progress()

        self.signal_mixture_created.emit(list_gaussian_first, list_gaussian_second,
                                         list_open3d_first, list_open3d_second)

    def update_progress(self):
        self.current_progress += 1
        new_percent = int(self.current_progress / self.max_progress * 100)
        self.signal_update_progress.emit(new_percent)

    def cancel(self):
        self.signal_cancel = True
