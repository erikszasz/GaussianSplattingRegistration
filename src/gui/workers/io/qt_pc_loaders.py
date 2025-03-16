from src.gui.workers.qt_base_worker import BaseWorker
from src.utils.file_loader import load_sparse_pc, load_o3d_pc, save_point_clouds_to_cache, \
    load_gaussian_pc


class PointCloudLoaderInput(BaseWorker):
    class ResultData:
        def __init__(self, point_cloud_first, point_cloud_second):
            self.point_cloud_first = point_cloud_first
            self.point_cloud_second = point_cloud_second

    def __init__(self, point_cloud_path_first, point_cloud_path_second):
        super().__init__()
        self.point_cloud_path_first = point_cloud_path_first
        self.point_cloud_path_second = point_cloud_path_second

    def run(self):
        result_first = load_sparse_pc(self.point_cloud_path_first)
        self.signal_progress.emit(50)
        result_second = load_sparse_pc(self.point_cloud_path_second)
        self.signal_progress.emit(100)
        self.signal_result.emit(PointCloudLoaderInput.ResultData(result_first, result_second))
        self.signal_finished.emit()


class PointCloudLoaderGaussian(BaseWorker):
    class ResultData:
        def __init__(self, o3d_point_cloud_first, o3d_point_cloud_second,
                     gaussian_point_cloud_first, gaussian_point_cloud_second):
            self.o3d_point_cloud_first = o3d_point_cloud_first
            self.o3d_point_cloud_second = o3d_point_cloud_second
            self.gaussian_point_cloud_first = gaussian_point_cloud_first
            self.gaussian_point_cloud_second = gaussian_point_cloud_second

    def __init__(self, point_cloud_path_first, point_cloud_path_second):
        super().__init__()
        self.point_cloud_path_first = point_cloud_path_first
        self.point_cloud_path_second = point_cloud_path_second

    def run(self):
        o3d_pc1, gs_pc1 = load_gaussian_pc(self.point_cloud_path_first)
        self.signal_progress.emit(50)
        o3d_pc2, gs_pc2 = load_gaussian_pc(self.point_cloud_path_second)
        self.signal_progress.emit(100)
        self.signal_result.emit(PointCloudLoaderGaussian.ResultData(o3d_pc1, o3d_pc2, gs_pc1, gs_pc2))
        self.signal_finished.emit()


class PointCloudLoaderO3D(BaseWorker):
    class ResultData:
        def __init__(self, point_cloud_first, point_cloud_second):
            self.point_cloud_first = point_cloud_first
            self.point_cloud_second = point_cloud_second

    def __init__(self, point_cloud1, point_cloud2):
        super().__init__()
        self.point_cloud1 = point_cloud1
        self.point_cloud2 = point_cloud2

    def run(self):
        result1 = load_o3d_pc(self.point_cloud1)
        self.signal_progress.emit(50)
        result2 = load_o3d_pc(self.point_cloud2)
        self.signal_progress.emit(100)
        self.signal_result.emit(PointCloudLoaderO3D.ResultData(result1, result2))
        self.signal_finished.emit()


class PointCloudSaver(BaseWorker):

    def __init__(self, point_cloud1, point_cloud2):
        super().__init__()
        self.point_cloud1 = point_cloud1
        self.point_cloud2 = point_cloud2

    def run(self):
        save_point_clouds_to_cache(self.point_cloud1, self.point_cloud2)
        self.signal_progress.emit(100)
        self.signal_finished.emit()
