from dataclasses import dataclass

import numpy as np
import open3d as o3d

from models.gaussian_model import GaussianModel


@dataclass
class PointCloudLoadParams:
    o3d_pc1: o3d.geometry.PointCloud
    o3d_pc2: o3d.geometry.PointCloud
    gauss_pc1: GaussianModel = None
    gauss_pc2: GaussianModel = None
    reset_view_point: bool = False

    keep_view: bool = False
    transformation_matrix: np.ndarray | None = None
    debug_color1: np.ndarray | None = None
    debug_color2: np.ndarray | None = None
