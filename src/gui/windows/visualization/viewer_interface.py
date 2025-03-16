import numpy as np
from PySide6.QtWidgets import QWidget

from params.io_parameters import PointCloudState


class ViewerInterface(QWidget):
    def on_embed_button_pressed(self):
        raise NotImplementedError

    def load_point_clouds(self, params: PointCloudState):
        raise NotImplementedError

    def get_current_view(self):
        raise NotImplementedError

    def get_camera_model(self):
        raise NotImplementedError

    def apply_camera_view(self, transformation):
        raise NotImplementedError

    @property
    def get_aabb(self):
        raise NotImplementedError

    def get_current_view_inner(self, extrinsic, tan_half_fov):
        aabb = self.get_aabb
        up = -extrinsic[1:2, 0:3].transpose()
        front = -extrinsic[2:3, 0:3].transpose()
        eye = np.linalg.inv(extrinsic[0:3, 0:3]) @ (extrinsic[0:3, 3:4] * -1.0)

        fov_rad = np.arctan(tan_half_fov) * 2.0
        fov = fov_rad * 180.0 / np.pi
        fov = np.max([np.min([fov, 90.0]), 50.0])

        bb_center = aabb.get_center()
        subtracted = (eye.reshape(3) - bb_center)
        ideal_distance = np.abs(subtracted.dot(front))[0]
        ideal_zoom = ideal_distance * np.tan(fov * 0.5 / 180.0 * np.pi) / aabb.get_max_extent()

        zoom = max([min([ideal_zoom, 2.0]), 0.02])
        view_ratio = zoom * aabb.get_max_extent()
        distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
        lookat = eye - front * distance

        return zoom, front.flatten(), lookat.flatten(), up.flatten()
