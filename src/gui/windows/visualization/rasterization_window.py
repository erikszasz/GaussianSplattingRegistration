import numpy as np
import torch
from PySide6 import QtCore
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, \
    QGraphicsSceneMouseEvent, QMainWindow, QWidget, QSizePolicy

from gui.windows.visualization.fx.temporal_anit_aliasing import TemporalAntiAliasing
from gui.windows.visualization.viewer_interface import ViewerInterface
from params.io_parameters import PointCloudLoadParams
from src.models.gaussian_model import GaussianModel
from src.utils.rasterization_util import rasterize_image, get_pixmap_from_tensor


class State:
    NONE = 0
    ROTATE = 1
    TRANSLATE = 2
    ROLL = 3
    ZOOM = 4


class PopupWindow(QMainWindow):
    def __init__(self, parent, camera):
        super().__init__(parent)

        self.camera = camera

        self.setWindowTitle("3DGS visualizer")

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        central_widget = QWidget(self)
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        self.graphics_view = None

    def set_graphics_view(self, graphics_view: QGraphicsView):
        if self.graphics_view is not None:
            self.layout.removeWidget(self.graphics_view)
            self.graphics_view.setParent(None)

        if graphics_view is not None:
            self.graphics_view = graphics_view
            self.layout.addWidget(self.graphics_view)
            self.graphics_view.setParent(self)
            self.graphics_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        new_width = self.width() - 20
        new_height = self.height() - 20
        self.camera.width = new_width
        self.camera.height = new_height
        self.graphics_view.setGeometry(0, 0, self.width(), self.height())


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
        self.graphics_view: QGraphicsView = None
        self.scene: QGraphicsScene = None
        self.pixmap_item: QGraphicsPixmapItem = None

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_view)

        # Mouse state variables
        self.mouse_down_x = 0
        self.mouse_down_y = 0
        self.state = State.NONE
        self.original_rotation = None
        self.original_position = None
        self.aabb = None

        self.rotation_speed = 0.01
        self.roll_speed = 0.1
        self.translate_speed = 7
        self.zoom_factor = 0.01

        # Approximate background color of the qdarkstyle theme
        self.background_color = np.array((0.09803921568627451, 0.13725490196078433, 0.17647058823529413))

        self.taa = TemporalAntiAliasing(0.3, 0.5)

        self.init_ui()

        self.is_embedded = True
        self.popup_window = None

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.graphics_view.setMouseTracking(False)
        self.graphics_view.setScene(self.scene)

        self.layout.addWidget(self.graphics_view)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.scene.mousePressEvent = self.mousePressEventScene
        self.scene.mouseMoveEvent = self.mouseMoveEventScene
        self.scene.mouseReleaseEvent = self.mouseReleaseEventScene
        self.scene.wheelEvent = self.wheelEventScene

        self.set_background_color(self.background_color)

    def set_background_color(self, rgb_array):
        r_255 = int(rgb_array[0] * 255)
        g_255 = int(rgb_array[1] * 255)
        b_255 = int(rgb_array[2] * 255)

        self.background_color = rgb_array
        self.graphics_view.setStyleSheet(f'background-color: rgb({r_255}, {g_255}, {b_255})')

    def mousePressEventScene(self, event: QGraphicsSceneMouseEvent):
        self.mouse_down_x = event.screenPos().x()
        self.mouse_down_y = event.screenPos().y()

        self.original_rotation = self.camera.rotation.clone()
        self.original_position = self.camera.position.clone()

        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.state = State.ROLL  # Shift + Left Button: Roll
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.state = State.TRANSLATE  # Ctrl + Left Button: Translate
            else:
                self.state = State.ROTATE  # Left Button: Rotate

        elif event.button() == Qt.MouseButton.MiddleButton:
            self.state = State.TRANSLATE  # Middle Button: Translate

    def mouseMoveEventScene(self, event: QGraphicsSceneMouseEvent):
        if self.state == State.NONE:
            return

        dx = event.screenPos().x() - self.mouse_down_x
        dy = event.screenPos().y() - self.mouse_down_y

        self.camera.rotation = self.original_rotation.clone()
        self.camera.position = self.original_position.clone()

        if self.state == State.ROTATE:
            self.camera.rotate(dx * self.rotation_speed, dy * self.rotation_speed)
        elif self.state == State.TRANSLATE:
            self.camera.translate(dx * self.translate_speed, dy * self.translate_speed)
        elif self.state == State.ROLL:
            self.camera.roll(dx * self.roll_speed)

    def mouseReleaseEventScene(self, event: QGraphicsSceneMouseEvent):
        self.state = State.NONE
        torch.cuda.empty_cache()

    def wheelEventScene(self, event):
        delta = event.delta()
        self.camera.zoom(delta * self.zoom_factor, self.get_aabb)

    @property
    def get_aabb(self):
        return self.aabb

    def update_view(self):
        if self.point_cloud_merged is None:
            return

        if self.camera is None:
            return

        image_tensor = rasterize_image(self.point_cloud_merged, self.camera, 1, self.background_color, "cuda:0", False)
        pix = get_pixmap_from_tensor(image_tensor)
        pix = self.taa.apply_taa(pix)
        self.pixmap_item.setPixmap(pix)
        self.scene.setSceneRect(self.pixmap_item.pixmap().rect())

    def set_active(self, active):
        if active:
            self.point_cloud_merged.move_to_device("cuda:0")
            self.timer.start(1)
            return

        self.timer.stop()
        if self.point_cloud_merged is not None:
            self.point_cloud_merged.move_to_device("cpu")
            torch.cuda.empty_cache()

    def on_embed_button_pressed(self):
        if not self.is_embedded:
            # Move back to main window
            if self.popup_window:
                self.popup_window.set_graphics_view(None)
                self.layout.addWidget(self.graphics_view)
                self.graphics_view.setParent(self)
                self.popup_window.close()
                self.is_embedded = True
                self.camera.height = self.height() - 20
                self.camera.width = self.width() - 20
        else:
            # Create and show the pop-up visualizer window
            if not self.popup_window:
                self.popup_window = PopupWindow(self, self.camera)

            self.popup_window.setGeometry(0, 0, self.camera.width + 20, self.camera.height + 20)
            self.popup_window.set_graphics_view(self.graphics_view)
            self.is_embedded = False
            self.popup_window.show()

    def update_transform(self, transformation):
        if self.pc1 is None or self.pc2 is None:
            return

        if self.point_cloud_merged is not None:
            self.point_cloud_merged.move_to_device("cpu")
            del self.point_cloud_merged
            torch.cuda.empty_cache()

        self.point_cloud_merged = GaussianModel.get_merged_gaussian_point_clouds(self.pc1, self.pc2, transformation)

    def load_point_clouds(self, params: PointCloudLoadParams):
        if self.point_cloud_merged is not None:
            self.point_cloud_merged.move_to_device("cpu")
            del self.point_cloud_merged
            torch.cuda.empty_cache()

        if self.pc1 is not None:
            del self.pc1

        if self.pc2 is not None:
            del self.pc2

        self.pc1 = params.gauss_pc1
        self.pc2 = params.gauss_pc2

        self.point_cloud_merged = GaussianModel.get_merged_gaussian_point_clouds(self.pc1, self.pc2,
                                                                                 params.transformation_matrix)

    def get_current_view(self):
        if self.camera is None:
            return

        extrinsics = self.camera.viewmat[0].detach().cpu().numpy()
        tan_half_fov = self.camera.height / (self.camera.intrinsics[0, 1, 1].item() * 2.0)
        return self.get_current_view_inner(extrinsics, tan_half_fov)

    def get_camera_model(self):
        return self.camera

    def apply_camera_view(self, transformation):
        if self.camera is None:
            return

        self.camera.set_viewmat(transformation)
