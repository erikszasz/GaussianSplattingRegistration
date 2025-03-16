#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from plyfile import PlyElement, PlyData

from src.models.gaussian_mixture_level import GaussianMixtureModel
from src.utils.general_utils import build_scaling_rotation, strip_symmetric, \
    inverse_sigmoid, matrices_to_quaternions, rebuild_lowerdiag, matrix_to_quaternion


class GaussianModel:

    def __init__(self, sh_degree: int = 3, device_name="cpu"):
        self.sh_degree = sh_degree
        self.device_name = device_name
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._covariance = torch.empty(0)

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symmetric = strip_symmetric(actual_covariance)
            return symmetric

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_colors(self):
        return self._features_dc.flatten(start_dim=1)

    @property
    def get_spherical_harmonics(self):
        return self._features_rest.flatten(start_dim=1)

    @property
    def get_opacity_with_activation(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_raw_opacity(self):
        return self._opacity

    def get_full_covariance(self, scaling_modifier=1.0):
        full_covariance = rebuild_lowerdiag(self._covariance)
        if scaling_modifier == 1:
            return full_covariance

        transformation_matrix = torch.diag_embed(torch.tensor([scaling_modifier] * 3)).to(self._covariance.device)
        return transformation_matrix @ full_covariance @ transformation_matrix.T

    def get_covariance(self, scaling_modifier=1):
        if scaling_modifier == 1:
            return self._covariance

        transformation_matrix = torch.diag_embed(torch.tensor([scaling_modifier] * 3)).to(self._covariance.device)
        transformed_covariances = (transformation_matrix @ self.get_full_covariance(scaling_modifier) @
                                   transformation_matrix.T)
        return strip_symmetric(transformed_covariances)

    def from_ply(self, plydata):
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coefficients) to (P, F, SH_coefficients except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device=self.device_name)
        self._features_dc = (torch.tensor(features_dc, dtype=torch.float, device=self.device_name)
                             .transpose(1, 2).contiguous())
        self._features_rest = (torch.tensor(features_extra, dtype=torch.float, device=self.device_name).
                               transpose(1, 2).contiguous())
        self._opacity = torch.tensor(opacities, dtype=torch.float, device=self.device_name)
        self._scaling = torch.tensor(scales, dtype=torch.float, device=self.device_name)
        self._rotation = torch.tensor(rots, dtype=torch.float, device=self.device_name)
        self._covariance = self.covariance_activation(self.get_scaling, 1.0, self._rotation)

    def from_mixture(self, gaussian_mixture: GaussianMixtureModel):
        self._xyz = torch.tensor(gaussian_mixture.xyz, dtype=torch.float, device=self.device_name)
        self._features_dc = (torch.tensor(gaussian_mixture.colors, dtype=torch.float, device=self.device_name)
                             .view(-1, 1, 3))
        self._features_rest = (torch.tensor(gaussian_mixture.features, dtype=torch.float, device=self.device_name)
                               .view(-1, (self.sh_degree + 1) ** 2 - 1, 3))
        self._opacity = torch.tensor(gaussian_mixture.opacities, dtype=torch.float, device=self.device_name)
        self._covariance = torch.tensor(gaussian_mixture.covariance, dtype=torch.float, device=self.device_name)

        eigenvalues, eigenvectors = self.decompose_covariance_matrix()
        self._scaling = eigenvalues
        self._rotation = matrices_to_quaternions(eigenvectors)

    def construct_list_of_attributes(self):
        attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attribute_list.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attribute_list.append('f_rest_{}'.format(i))
        attribute_list.append('opacity')
        for i in range(self._scaling.shape[1]):
            attribute_list.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            attribute_list.append('rot_{}'.format(i))
        return attribute_list

    def save_ply(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        plydata = PlyData([el])
        plydata.write(path)

    def clone_gaussian(self):
        new_model = GaussianModel(3)
        new_model._covariance = self._covariance.clone().detach()
        new_model._xyz = self._xyz.clone().detach()
        new_model._rotation = self._rotation.clone().detach()
        new_model._scaling = self._scaling.clone().detach()
        new_model._features_dc = self._features_dc.clone().detach()
        new_model._features_rest = self._features_rest.clone().detach()
        new_model._opacity = self._opacity.clone().detach()
        return new_model

    def quat_multiply(self, quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.chunk(quaternion0, 4, dim=-1)
        w1, x1, y1, z1 = torch.chunk(quaternion1, 4, dim=-1)

        return torch.cat((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ), dim=-1)

    def transform_gaussian_model(self, transformation_matrix):
        rotation_matrix = transformation_matrix[:3, :3]
        self._xyz = torch.matmul(self._xyz, rotation_matrix.T)
        self._xyz += transformation_matrix[:3, 3]

        transformed_covariances = rotation_matrix @ self.get_full_covariance() @ rotation_matrix.transpose(
            0, 1)
        self._covariance = strip_symmetric(transformed_covariances)

        quaternions = matrix_to_quaternion(rotation_matrix).unsqueeze(0).to(self._rotation.device)
        rotations_from_quats = self.quat_multiply(self._rotation, quaternions)
        self._rotation = rotations_from_quats / torch.norm(rotations_from_quats, p=2, dim=-1, keepdim=True)
        pass

    def move_to_device(self, device_name):
        if self.device_name == device_name:
            return

        self.device_name = device_name
        self._xyz = self._xyz.to(device_name)
        self._features_dc = self._features_dc.to(device_name)
        self._features_rest = self._features_rest.to(device_name)
        self._scaling = self._scaling.to(device_name)
        self._rotation = self._rotation.to(device_name)
        self._opacity = self._opacity.to(device_name)
        self._covariance = self._covariance.to(device_name)

    """
    Executes eigendecomposition of the covariance matrix. The function is not used, but left in for completeness.
    Hopefully no one wants to save a downscaled gaussian model.
    """

    def decompose_covariance_matrix(self):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.get_full_covariance())

        # Standard basis vectors for x, y, z axes
        x_axis = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device_name)
        y_axis = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device_name)
        z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device_name)

        axes = torch.stack([x_axis, y_axis, z_axis])

        # Compute dot products to determine the correspondence of each eigenvector
        dot_products = torch.abs(torch.matmul(eigenvectors.transpose(1, 2), axes.T))

        # Get the indices that would sort each eigenvector to align with x, y, z axes
        correspondence = torch.argmax(dot_products, dim=2)

        # Reorder eigenvalues and eigenvectors accordingly
        sorted_eigenvalues = torch.zeros_like(eigenvalues)
        sorted_eigenvectors = torch.zeros_like(eigenvectors)

        sorted_eigenvalues.scatter_(1, correspondence, eigenvalues)
        sorted_eigenvectors.scatter_(1, correspondence.unsqueeze(-1).expand(-1, -1, 3), eigenvectors)

        return sorted_eigenvalues, sorted_eigenvectors

    @staticmethod
    def get_merged_gaussian_point_clouds(gaussian1, gaussian2, transformation_matrix):
        merged_pc = GaussianModel(3)
        gaussian1_copy = gaussian1

        # If the transformation matrix is not an identity matrix
        if (transformation_matrix is not None and
                not np.array_equal(transformation_matrix, np.eye(transformation_matrix.shape[0]))):
            transformation_matrix_tensor = (torch.from_numpy(transformation_matrix.astype(np.float32))
                                            .to(gaussian1.device_name))
            gaussian1_copy = gaussian1.clone_gaussian()
            gaussian1_copy.transform_gaussian_model(transformation_matrix_tensor)

        merged_pc._xyz = torch.cat((gaussian1_copy._xyz, gaussian2._xyz))
        merged_pc._rotation = torch.cat((gaussian1_copy._rotation, gaussian2._rotation))
        merged_pc._scaling = torch.cat((gaussian1_copy._scaling, gaussian2._scaling))
        merged_pc._features_dc = torch.cat((gaussian1_copy._features_dc, gaussian2._features_dc))
        merged_pc._features_rest = torch.cat((gaussian1_copy._features_rest, gaussian2._features_rest))
        merged_pc._opacity = torch.cat((gaussian1_copy._opacity, gaussian2._opacity))
        merged_pc._covariance = torch.cat((gaussian1_copy._covariance, gaussian2._covariance))

        return merged_pc
