import torch
from PIL.ImageQt import ImageQt
from PySide6 import QtGui
from gsplat.rendering import rasterization
import torchvision.transforms.functional as F

from src.models.gaussian_model import GaussianModel


def rasterize_image(point_cloud: GaussianModel, camera, scale, color, device, leave_on_gpu=True, ):
    color_tensor = torch.tensor(color, dtype=torch.float32, device=device).view(1, -1)
    covars = point_cloud.get_full_covariance(scale)
    render_colors, _, _ = rasterization(
        point_cloud.get_xyz,
        point_cloud.get_rotation,
        point_cloud.get_scaling,
        point_cloud.get_opacity_with_activation.view(-1),
        point_cloud.get_features,
        camera.viewmat.cuda(),
        camera.intrinsics.cuda(),
        camera.width,
        camera.height,
        render_mode="RGB",
        sh_degree=point_cloud.sh_degree,
        backgrounds=color_tensor,
        covars=covars,
        packed=True,
        radius_clip=3
    )

    return render_colors.detach() if leave_on_gpu else render_colors.cpu()


def get_pixmap_from_tensor(image_tensor):
    img_save = F.to_pil_image(image_tensor[0].permute(2, 0, 1).clamp(0, 1).detach().cpu())
    qim = ImageQt(img_save)
    pix = QtGui.QPixmap.fromImage(qim)
    return pix
