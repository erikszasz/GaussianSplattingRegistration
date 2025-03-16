from dataclasses import dataclass

import numpy as np
import open3d as o3d

from models.gaussian_model import GaussianModel


@dataclass
class PointCloudState:
    o3d_pc1: o3d.geometry.PointCloud
    o3d_pc2: o3d.geometry.PointCloud
    gauss_pc1: GaussianModel = None
    gauss_pc2: GaussianModel = None
    reset_view_point: bool = False

    keep_view: bool = False
    transformation_matrix: np.ndarray | None = None
    debug_color1: np.ndarray | None = None
    debug_color2: np.ndarray | None = None


@dataclass
class LoadRequestParams:
    first_path: str
    second_path: str

    save_converted: bool = False


@dataclass
class SaveRequestParams:
    save_path: str

    transformation_matrix: np.ndarray | None = None
    use_corresponding_pc: bool = False
    first_path: str | None = None
    second_path: str | None = None
