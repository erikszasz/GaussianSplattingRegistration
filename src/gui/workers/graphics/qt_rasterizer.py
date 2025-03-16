import torch

from src.gui.workers.qt_base_worker import BaseWorker
from src.models.gaussian_model import GaussianModel
from src.utils.rasterization_util import rasterize_image, get_pixmap_from_tensor


class RasterizerWorker(BaseWorker):

    def __init__(self, pc1, pc2, transformation, camera, scale, color):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.pc1 = pc1
        self.pc2 = pc2

        self.transformation = transformation
        self.camera = camera
        self.scale = scale
        self.color = color

    def run(self):
        with torch.no_grad():
            point_cloud = GaussianModel.get_merged_gaussian_point_clouds(self.pc1, self.pc2, self.transformation)

            point_cloud.move_to_device(self.device)
            image_tensor = rasterize_image(point_cloud, self.camera, self.scale, self.color, self.device, False)
            pix = get_pixmap_from_tensor(image_tensor)

            del point_cloud
            torch.cuda.empty_cache()
            self.signal_result.emit(pix)

        self.signal_progress.emit(100)
        #self.signal_finished.emit()
        torch.cuda.empty_cache()


