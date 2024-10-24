import numpy as np
import torch
from PySide6 import QtCore
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QScrollArea, QSizePolicy

from gui.windows.visualization.viewer_interface import ViewerInterface
from src.models.gaussian_model import GaussianModel
from src.utils.rasterization_util import rasterize_image, get_pixmap_from_tensor


# noinspection PyTypeChecker
class GaussianSplatWindow(ViewerInterface):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Viewer')

        self.pc1 = None
        self.pc2 = None
        self.point_cloud_merged = None
        self.camera = None

        self.layout: QVBoxLayout = None
        self.scroll_area = None
        self.render_label: QLabel = None

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_view)

        self.last_mouse_position = None
        self.left_mouse_pressed = False
        self.setMouseTracking(True)

        self.rotation_speed = np.radians(1)
        self.zoom_factor = 0.01
        self.speed = 0.01

        # Approximate background color of the qdarkstyle theme
        self.background_color = np.array((0.09803921568627451, 0.13725490196078433, 0.17647058823529413))

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.render_label = QLabel()
        self.render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_label.setScaledContents(False)
        self.render_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.scroll_area.setWidget(self.render_label)
        self.layout.addWidget(self.scroll_area)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.set_background_color(self.background_color)

    def resizeEvent(self, event):
        if self.render_label.pixmap() is None:
            return

        scroll_area_size = self.scroll_area.viewport().size()
        scaled_pixmap = self.render_label.pixmap().scaled(scroll_area_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.render_label.setPixmap(scaled_pixmap)
        self.render_label.resize(scaled_pixmap.size())

        super().resizeEvent(event)

    def set_background_color(self, rgb_array):
        r_255 = int(rgb_array[0] * 255)
        g_255 = int(rgb_array[1] * 255)
        b_255 = int(rgb_array[2] * 255)

        self.background_color = rgb_array
        self.render_label.setStyleSheet(f'background-color: rgb({r_255}, {g_255}, {b_255})')

    def mouseMoveEvent(self, event):
        if self.last_mouse_position is None or not self.left_mouse_pressed:
            return

        if self.camera is None:
            return

        dx = self.last_mouse_position.x() - event.x()
        dy = self.last_mouse_position.y() - event.y()

        self.camera.rotate(dx * self.rotation_speed, dy * self.rotation_speed)

        self.last_mouse_position = event.pos()
        self.camera.update_view_matrix()
        self.update_view()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.camera is None:
            return

        self.left_mouse_pressed = True
        self.last_mouse_position = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.camera is None:
            return

        self.left_mouse_pressed = False

    def wheelEvent(self, event):
        if self.camera is None:
            return

        delta = event.angleDelta().y()
        self.camera.zoom(delta * self.zoom_factor)
        self.camera.update_view_matrix()
        self.update_view()

    def update_view(self):
        if self.point_cloud_merged is None:
            return

        if self.camera is None:
            return

        image_tensor = rasterize_image(self.point_cloud_merged, self.camera, 1, self.background_color, "cuda:0", False)
        pix = get_pixmap_from_tensor(image_tensor)
        self.render_label.setPixmap(pix)

    def set_active(self, active):
        if active:
            self.point_cloud_merged.move_to_device("cuda:0")
            self.timer.start(30)
            return

        self.timer.stop()
        if self.point_cloud_merged is not None:
            self.point_cloud_merged.move_to_device("cpu")
            torch.cuda.empty_cache()

    # TODO: Implement if needed
    def on_embed_button_pressed(self):
        pass

    def update_transform(self, transformation):
        if self.pc1 is None or self.pc2 is None:
            return

        if self.point_cloud_merged is not None:
            self.point_cloud_merged.move_to_device("cpu")
            del self.point_cloud_merged
            torch.cuda.empty_cache()

        self.point_cloud_merged = GaussianModel.get_merged_gaussian_point_clouds(self.pc1, self.pc2, transformation)

    def load_point_clouds(self, pc1, pc2, transformation):
        if self.point_cloud_merged is not None:
            self.point_cloud_merged.move_to_device("cpu")
            del self.point_cloud_merged
            torch.cuda.empty_cache()

        if self.pc1 is not None:
            del self.pc1

        if self.pc2 is not None:
            del self.pc2

        self.pc1 = pc1
        self.pc2 = pc2

        self.point_cloud_merged = GaussianModel.get_merged_gaussian_point_clouds(self.pc1, self.pc2, transformation)

    def get_current_view(self, aabb):
        if self.camera is None:
            return

        extrinsics = self.camera.viewmat[0].detach().cpu().numpy()
        tan_half_fov = self.camera.height / (self.camera.intrinsics[0, 1, 1].item() * 2.0)
        return self.get_current_view_inner(extrinsics, tan_half_fov, aabb)

    def get_camera_model(self):
        return self.camera

    def apply_camera_view(self, transformation):
        if self.camera is None:
            return

        self.camera.set_viewmat(transformation)

