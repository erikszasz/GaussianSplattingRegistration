import torch
from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QStackedWidget
import open3d as o3d

from params.io_parameters import PointCloudLoadParams
from src.gui.windows.visualization.rasterization_window import GaussianSplatWindow
from src.gui.windows.visualization.open3d_window import Open3DWindow
from utils.plane_fitting_util import get_o3d_plane


class VisualizerWindow(QWidget):
    def __init__(self, parent):
        super(VisualizerWindow, self).__init__()
        self.parent_window = parent
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setBaseSize(self.maximumSize())

        self.vis_stack = QStackedWidget()
        self.vis_open3d = Open3DWindow(self)
        self.vis_3dgs = GaussianSplatWindow()
        self.vis_stack.addWidget(self.vis_open3d)
        self.vis_stack.addWidget(self.vis_3dgs)
        self.vis_stack.setCurrentIndex(0)

        self.layout.addWidget(self.vis_stack)
        self.vis_open3d.set_active(True)

        self.o3d_pc1 = self.vis_open3d.pc1
        self.o3d_pc2 = self.vis_open3d.pc2
        self.gauss_pc1 = None
        self.gauss_pc2 = None

    @property
    def get_camera(self):
        if self.vis_stack.currentIndex() == 0:
            return self.vis_open3d.get_camera_model()

        return self.vis_3dgs.get_camera_model()

    def load_point_clouds(self, params: PointCloudLoadParams):
        self.o3d_pc1 = self.o3d_pc1
        self.o3d_pc2 = self.o3d_pc2
        self.gauss_pc2 = self.gauss_pc2
        self.gauss_pc1 = self.gauss_pc1

        self.vis_open3d.set_active(False)
        self.vis_3dgs.set_active(False)

        self.vis_open3d.load_point_clouds(params)

        if self.gauss_pc1 is not None:
            self.vis_3dgs.load_point_clouds(params)

        self.vis_open3d.set_active(True) if self.vis_stack.currentIndex() == 0 else self.vis_3dgs.set_active(True)

        if params.reset_view_point:
            self.reset_view_point()

    def on_embed_button_pressed(self):
        if self.vis_stack.currentIndex() == 0:
            self.vis_open3d.on_embed_button_pressed()
            return

        self.vis_3dgs.on_embed_button_pressed()

    def vis_type_changed(self, index):
        if index == 0:
            self.vis_3dgs.set_active(False)

            # If the o3d camera is orthogonal, there is no need for a camera update, because 3DGS view is not allowed.
            # Due to this, the camera pose could not change.
            if not self.vis_open3d.is_ortho():
                self.vis_open3d.update_camera_view(self.get_camera)

            self.vis_open3d.set_active(True)
        elif self.vis_open3d.is_ortho():
            return
        else:
            self.vis_open3d.set_active(False)
            self.vis_3dgs.camera = self.get_camera
            self.vis_3dgs.aabb = self.vis_open3d.get_aabb
            self.vis_3dgs.set_active(True)

        self.vis_stack.setCurrentIndex(index)

    def update_transform(self, transformation, dc1, dc2):
        self.vis_open3d.set_active(False)
        self.vis_3dgs.set_active(False)

        self.vis_open3d.update_transform(transformation, dc1, dc2)
        self.vis_3dgs.update_transform(transformation)

        if self.vis_stack.currentIndex() == 0:
            self.vis_open3d.set_active(True)
            return

        self.vis_3dgs.set_active(True)

    def update_visualizer_settings_o3d(self, zoom, front, lookat, up):
        self.vis_open3d.update_visualizer(zoom, front, lookat, up)

    def update_visualizer_settings_3dgs(self, zoom, front, lookat, up):
        self.update_visualizer_settings_o3d(zoom, front, lookat, up)
        self.vis_3dgs.set_active(False)
        self.vis_3dgs.apply_camera_view(torch.tensor(self.vis_open3d.get_camera_extrinsic(), dtype=torch.float32))
        self.vis_3dgs.set_active(True)

    def get_current_view(self):
        if self.vis_stack.currentIndex() == 0:
            return self.vis_open3d.get_current_view()

        return self.vis_3dgs.get_current_view()

    def is_ortho(self):
        if self.vis_stack.currentIndex() == 0:
            return self.vis_open3d.is_ortho()

        return False

    def apply_camera_view(self, transformation):
        self.vis_open3d.set_active(False)
        self.vis_3dgs.set_active(False)

        self.vis_open3d.apply_camera_view(transformation)
        self.vis_3dgs.apply_camera_view(transformation)

        if self.vis_stack.currentIndex() == 0:
            self.vis_open3d.set_active(True)
            return

        self.vis_3dgs.set_active(True)

    def reset_view_point(self):
        self.vis_open3d.set_active(False)
        self.vis_3dgs.set_active(False)

        self.vis_open3d.vis.reset_view_point(True)
        self.vis_3dgs.apply_camera_view(torch.tensor(self.vis_open3d.get_camera_extrinsic(), dtype=torch.float32))

        if self.vis_stack.currentIndex() == 0:
            self.vis_open3d.set_active(True)
            return

        self.vis_3dgs.set_active(True)

    def add_plane(self, plane_coefficient, pos_tensor, indices, color, pc_index):
        # TODO: Visualize plane in raster mode
        if self.vis_stack.currentIndex() != 0:
            return

        self.vis_open3d.paint_inliers(indices, color, pc_index)
        self.vis_open3d.add_plane(get_o3d_plane(plane_coefficient, pos_tensor, color))

