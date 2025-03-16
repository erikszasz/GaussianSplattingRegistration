import numpy as np
import torch

from src.utils.graphics_utils import getWorld2View2, getView2World2
from utils.math_util import axis_angle_rotation


class Camera:
    right = torch.tensor([1, 0, 0], dtype=torch.float32)
    up = torch.tensor([0, 1, 0], dtype=torch.float32)
    forward = torch.tensor([0, 0, 1], dtype=torch.float32)

    def __init__(self, R, T, fx, fy, image_name, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(Camera, self).__init__()
        self.image_name = image_name
        self.position = torch.tensor(T, dtype=torch.float32)
        self.rotation = torch.tensor(R, dtype=torch.float32)
        self.width = width
        self.height = height
        self.intrinsics = torch.tensor([
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1]
        ], dtype=torch.float32)[None, :, :]

        # Initial view matrix
        self.viewmat = torch.tensor(getWorld2View2(R, T, trans, scale))[None, :, :]
        self.target = None

    # TODO: Implement trackball like rotation around center point
    def rotate(self, dx, dy):
        torch_yawn = torch.tensor(dx)
        torch_pitch = torch.tensor(-dy)

        yaw_rotation_matrix = axis_angle_rotation(self.get_up_vector(), torch_yawn)
        pitch_rotation_matrix = axis_angle_rotation(self.get_right_vector(), torch_pitch)

        self.rotation = yaw_rotation_matrix @ pitch_rotation_matrix @ self.rotation

        self.update_view_matrix()

    # TODO: Consider zoom
    def translate(self, dx, dy):
        world_move = self.calc_pan_vector_world(dx, dy)
        self.position += world_move

        self.update_view_matrix()

    # TODO: Roll around center
    def roll(self, dx):
        radians = self.calc_rotate_z_radians(dx)
        rotation_matrix = axis_angle_rotation(self.forward, radians)
        self.rotation = self.rotation @ rotation_matrix

        self.update_view_matrix()

    def get_forward_vector(self):
        forward = self.rotation @ self.forward
        return forward

    def get_up_vector(self):
        forward = self.rotation @ self.up
        return forward

    def get_right_vector(self):
        forward = self.rotation @ self.right
        return forward

    def calc_pan_vector_world(self, dx, dy):
        units_per_px_x = 1.0 / self.intrinsics[0, 0, 0]
        units_per_px_y = 1.0 / self.intrinsics[0, 1, 1]
        camera_local_move = torch.tensor([dx * units_per_px_x, dy * units_per_px_y, 0], dtype=torch.float32)
        world_move = (self.right * camera_local_move[0]) + (self.up * camera_local_move[1])

        return world_move

    def calc_rotate_z_radians(self, dx):
        return 4.0 * np.pi * dx / self.height

    def zoom(self, delta, aabb):
        if not self.can_zoom(delta, aabb):
            return

        model_size = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
        length = np.linalg.norm(aabb.get_center() - self.position.detach().numpy())
        length = max(0.02 * model_size, length)
        dist = delta * 0.05 * length
        self.position += dist * self.forward
        self.update_view_matrix()

    def can_zoom(self, delta, aabb):
        tan_half_fov = self.height / (self.intrinsics[0, 1, 1].item() * 2.0)
        fov_rad = np.arctan(tan_half_fov) * 2.0
        fov = fov_rad * 180.0 / np.pi
        fov = np.max([np.min([fov, 90.0]), 50.0])

        eye = np.linalg.inv(self.rotation.detach().numpy()) @ (self.position.detach().numpy() * -1.0)

        bb_center = aabb.get_center()
        subtracted = (eye.reshape(3) - bb_center)
        front = -self.rotation.detach().numpy()[2:3, 0:3].transpose()
        ideal_distance = np.abs(subtracted.dot(front))[0]
        ideal_zoom = ideal_distance * np.tan(fov * 0.5 / 180.0 * np.pi) / aabb.get_max_extent()

        if ideal_zoom > 2 and delta > 0:
            return False

        if ideal_zoom < 0.02 and delta < 0:
            return False

        return True

    def update_view_matrix(self):
        self.viewmat = torch.tensor(getWorld2View2(self.rotation.numpy(), self.position.numpy()))[None, :, :]

    def set_viewmat(self, tranformation):
        self.viewmat = tranformation[None, :, :]
        R, T = getView2World2(tranformation.detach().cpu().numpy())

        self.position = torch.tensor(T, dtype=torch.float32)
        self.rotation = torch.tensor(R, dtype=torch.float32)
