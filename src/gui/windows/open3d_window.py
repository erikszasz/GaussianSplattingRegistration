import copy

import numpy as np
import open3d as o3d
import win32gui
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow


class Open3DWindow(QMainWindow):
    def __init__(self):
        super(Open3DWindow, self).__init__()
        self.pc1_copy = None
        self.pc2_copy = None
        self.pc1 = None
        self.pc2 = None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        self.setCentralWidget(widget)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Set background color to match theme
        background_color = (0.09803921568627451, 0.13725490196078433, 0.17647058823529413)
        opt = self.vis.get_render_option()
        opt.background_color = background_color

        # TODO: Find workaround for linux/mac
        hwnd = win32gui.FindWindowEx(0, 0, None, "Open3D")
        self.window = QtGui.QWindow.fromWinId(hwnd)
        self.window_container = self.createWindowContainer(self.window, widget)
        layout.addWidget(self.window_container, 0, 0)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(1)

        # Example data
        knot_mesh = o3d.data.KnotMesh()
        mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
        mesh.compute_vertex_normals()
        self.vis.add_geometry(mesh)

    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def load_point_clouds(self, point_cloud_first, point_cloud_second):
        self.pc1 = point_cloud_first
        self.pc2 = point_cloud_second
        self.pc1_copy = self.pc1
        self.pc2_copy = self.pc2

        self.vis.clear_geometries()
        self.vis.add_geometry(point_cloud_first)
        self.vis.add_geometry(point_cloud_second)

    def closeEvent(self, event):
        self.vis.destroy_window()
        super(QMainWindow, self).closeEvent(event)

    def update_transform(self, transformation):
        if not self.pc1 or not self.pc2:
            return

        source_temp = copy.deepcopy(self.pc1)
        target_temp = copy.deepcopy(self.pc2)
        target_temp.transform(transformation)

        self.vis.clear_geometries()
        self.vis.add_geometry(source_temp)
        self.vis.add_geometry(target_temp)

    def update_transform_with_colors(self, debug_color1, debug_color2, transformation):
        if not self.pc1 or not self.pc2:
            return

        source_temp = copy.deepcopy(self.pc1)
        target_temp = copy.deepcopy(self.pc2)

        source_temp.paint_uniform_color(debug_color1)
        target_temp.paint_uniform_color(debug_color2)

        target_temp.transform(transformation)

        self.vis.clear_geometries()
        self.vis.add_geometry(source_temp)
        self.vis.add_geometry(target_temp)

    def update_visualizer(self, zoom, front, lookat, up):
        view_control = self.vis.get_view_control()
        view_control.set_zoom(zoom)
        view_control.set_front(front)
        view_control.set_lookat(lookat)
        view_control.set_up(up)

    # TODO: Fix conversion error
    def get_current_view(self):
        view_control = self.vis.get_view_control()
        parameters = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = parameters.extrinsic

        right = extrinsic[0:1, 0:3].transpose()
        up = -extrinsic[1:2, 0:3].transpose()
        front = -extrinsic[2:3, 0:3].transpose()
        eye = np.linalg.inv(extrinsic[0:3, 0:3]) @ (extrinsic[0:3, 3:4] * -1.0)

        # Calculate the aab
        combined_vertices = np.vstack((np.asarray(self.pc1_copy.points), np.asarray(self.pc2_copy.points)))
        # Compute the minimum and maximum coordinates to find the extents of the AABB
        aabb_min = np.min(combined_vertices, axis=0)
        aabb_max = np.max(combined_vertices, axis=0)
        # Create a new AABB geometry
        aabb = o3d.geometry.AxisAlignedBoundingBox(aabb_min, aabb_max)

        fov = view_control.get_field_of_view()

        bb_center = aabb.get_center()
        subtracted = (eye - bb_center.reshape(3, 1)).T
        ideal_distance = np.abs(subtracted.dot(front))
        ideal_zoom = ideal_distance * np.tan(fov * 0.5 / 180.0 * np.pi) / aabb.get_max_extent()
        zoom = ideal_zoom
        zoom = max([min([ideal_zoom, 2.0]), 0.02])
        view_ratio = zoom * aabb.get_max_extent()
        distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
        lookat = eye - front * distance

        zoom_float = 0.0
        try:
            zoom_float = zoom.flatten()[0]
        except AttributeError:
            zoom_float = zoom
        return zoom_float, front.flatten(), lookat.flatten(), up.flatten()
