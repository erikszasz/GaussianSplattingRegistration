import json
import os.path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from PySide6 import QtWidgets

from src.gui.workers.qt_base_worker import BaseWorker
from src.models.gaussian_model import GaussianModel
from lpips import LPIPS
from src.utils.evaluation_utils import ssim, psnr, mse
from src.utils.rasterization_util import rasterize_image


class RegistrationEvaluator(BaseWorker):
    def __init__(self, pc1, pc2, transformation, cameras_list, images_path, log_path, color, registration_result,
                 use_gpu):

        super().__init__()
        # Signal to cancel task
        self.signal_cancel = False

        self.pc1 = pc1
        self.pc2 = pc2

        self.transformation = transformation

        self.cameras_list = cameras_list
        self.images_path = images_path
        self.log_path = log_path

        self.color = color
        self.use_gpu = use_gpu  # Whether the actual evaluation happens on the gpu or not
        self.device = "cuda:0"

        self.registration_result = registration_result
        self.mean_rmses = None
        self.mean_ssims = None
        self.mean_psnrs = None
        self.mean_lpipss = None
        self.mean_mses = None

        self.current_progress = 0
        self.max_progress = len(cameras_list)

    def run(self):
        point_cloud = GaussianModel.get_merged_gaussian_point_clouds(self.pc1, self.pc2, self.transformation)
        point_cloud.move_to_device(self.device)
        lpips_loss = LPIPS(net='alex').to(self.device)

        error_list = []
        mses = []
        rmses = []
        ssims = []
        psnrs = []
        lpipss = []

        for idx, camera in enumerate(self.cameras_list):
            # Process events, look for cancel signal
            QtWidgets.QApplication.processEvents()
            if self.signal_cancel:
                # Force gpu memory garbage collection
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                return

            self.update_progress()

            img_name = camera.image_name
            image_path = os.path.join(self.images_path, img_name + ".png")
            try:
                pil_image = Image.open(image_path).convert('RGB')
                gt_image = tf.to_tensor(pil_image).unsqueeze(0)
                if self.use_gpu:
                    gt_image = gt_image.cuda()
            except (OSError, IOError) as e:
                error_list.append(str(e))
                continue

            try:
                image_tensor = (rasterize_image(point_cloud, camera, 1, self.color, self.device, self.use_gpu)
                                .permute(0, 3, 1, 2))
            except (OSError, IOError) as e:
                error_list.append(str(e))
                continue

            current_mse = mse(image_tensor, gt_image)
            mses.append(current_mse)
            rmses.append(torch.sqrt(current_mse))
            ssims.append(ssim(image_tensor, gt_image))
            psnrs.append(psnr(image_tensor, gt_image))
            lpipss.append(lpips_loss(image_tensor, gt_image))

            del image_tensor, gt_image
            torch.cuda.empty_cache()

        self.mean_mses = torch.tensor(mses).mean().item()
        self.mean_rmses = torch.tensor(rmses).mean().item()
        self.mean_ssims = torch.tensor(ssims).mean().item()
        self.mean_psnrs = torch.tensor(psnrs).mean().item()
        self.mean_lpipss = torch.tensor(lpipss).mean().item()

        log = self.create_and_save_log_file(error_list)

        del point_cloud
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        self.signal_result.emit(log)
        self.signal_progress.emit(100)
        self.signal_finished.emit()

    def cancel_evaluation(self):
        self.signal_cancel = True

    def update_progress(self):
        self.current_progress += 1
        new_percent = int(self.current_progress / self.max_progress * 100)
        self.signal_progress.emit(new_percent)

    def create_and_save_log_file(self, error_list):
        evaluation = self.EvaluationObject(self.registration_result, self.mean_mses, self.mean_rmses,
                                           self.mean_ssims, self.mean_psnrs, self.mean_lpipss, error_list)
        with open(self.log_path, 'w') as f:
            json.dump(evaluation.__dict__, f, default=lambda x: x.tolist(), indent=2)

        return evaluation

    # Class for the JSON evaluation
    class EvaluationObject(object):

        registration_data: dict
        mse: float
        rmse: float
        ssim: float
        psnr: float
        lpips: float

        error_list: list

        def __init__(self, registration_data, mse, rmse, ssim, psnr, lpips, error_list):
            super().__init__()

            self.registration_data = dict()
            if registration_data is not None:
                self.registration_data = registration_data.__dict__
            self.mse = mse
            self.rmse = rmse
            self.ssim = ssim
            self.psnr = psnr
            self.lpips = lpips
            self.error_list = error_list

