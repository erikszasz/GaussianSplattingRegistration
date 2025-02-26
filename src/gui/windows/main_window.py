from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSplitter, QGroupBox, \
    QTabWidget, QErrorMessage, QMessageBox, QSizePolicy

from controllers.downsampler_controller import DownsamplerController
from controllers.plane_fitting_controller import PlaneFittingController
from controllers.point_cloud_io_controller import PointCloudIOController
from controllers.registration_controller import RegistrationController
from gui.tabs.plane_fitting_tab import PlaneFittingTab
from gui.workers.graphics.qt_rasterizer import RasterizerWorker
from models.data_repository import DataRepository
from models.ui_state_repository import UIStateRepository
from params.io_parameters import PointCloudState
from src.gui.tabs.evaluation_tab import EvaluationTab
from src.gui.tabs.gaussian_mixture_tab import GaussianMixtureTab
from src.gui.tabs.global_registration_tab import GlobalRegistrationTab
from src.gui.tabs.input_tab import InputTab
from src.gui.tabs.local_registration_tab import LocalRegistrationTab
from src.gui.tabs.merger_tab import MergeTab
from src.gui.tabs.multi_scale_registration_tab import MultiScaleRegistrationTab
from src.gui.tabs.rasterizer_tab import RasterizerTab
from src.gui.tabs.visualizer_tab import VisualizerTab
from src.gui.widgets.progress_dialog_factory import ProgressDialogFactory
from src.gui.widgets.transformation_widget import Transformation3DPicker
from src.gui.windows.visualization.image_viewer_window import RasterImageViewer
from src.gui.windows.visualizer_window import VisualizerWindow
from src.gui.workers.qt_base_worker import move_worker_to_thread
from src.models.camera import Camera
from src.utils.graphics_utils import get_focal_from_intrinsics


class RegistrationMainWindow(QMainWindow):

    # Constructor
    def __init__(self, parent=None):
        super(RegistrationMainWindow, self).__init__(parent)
        self.setWindowTitle("Gaussian Splatting Registration")
        # Set window size to screen size
        self.showMaximized()

        # Central data repository for handling the point clouds and their corresponding data
        self.data_repository = DataRepository()
        self.ui_repository = UIStateRepository()

        # Controllers
        self.io_controller = PointCloudIOController(self.data_repository, self.ui_repository)
        self.plane_fitting_controller = PlaneFittingController(self.data_repository, self.ui_repository)
        self.registration_controller = RegistrationController(self.data_repository, self.ui_repository)
        self.downsampler_controller = DownsamplerController(self.data_repository, self.ui_repository)

        # Tabs for the settings page
        self.visualizer_widget = None
        self.transformation_picker = None
        self.hem_widget = None

        # Image viewer
        self.raster_window = None

        # Create splitter and two planes
        splitter = QSplitter(self)

        self.visualizer_window = VisualizerWindow(self)
        pane_data = QWidget()

        layout_pane = QVBoxLayout()
        pane_data.setLayout(layout_pane)

        group_input_data = QGroupBox()
        self.setup_input_group(group_input_data)

        group_registration = QGroupBox()
        self.setup_registration_group(group_registration)

        layout_pane.addWidget(group_input_data)
        layout_pane.addWidget(group_registration)
        layout_pane.setStretch(0, 1)
        layout_pane.setStretch(1, 1)

        splitter.addWidget(self.visualizer_window)
        splitter.addWidget(pane_data)

        splitter.setOrientation(Qt.Orientation.Horizontal)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.setCentralWidget(splitter)

        self.register_controller_handlers()
        self.register_repository_handlers()

    # GUI setup functions
    def setup_input_group(self, group_input_data):
        group_input_data.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        group_input_data.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        group_input_data.setTitle("Inputs and settings")
        layout = QVBoxLayout()
        group_input_data.setLayout(layout)

        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        input_tab = InputTab()
        self.transformation_picker = Transformation3DPicker(self.ui_repository)
        self.visualizer_widget = VisualizerTab(self)
        rasterizer_tab = RasterizerTab()
        merger_widget = MergeTab(self.ui_repository)

        input_tab.signal_load_gaussian.connect(self.io_controller.handle_gaussian_load)
        input_tab.signal_load_sparse.connect(self.io_controller.handle_sparse_load)
        input_tab.signal_load_cached.connect(self.io_controller.handle_cached_load)
        self.visualizer_widget.signal_change_vis_settings_o3d.connect(self.change_visualizer_settings_o3d)
        self.visualizer_widget.signal_change_vis_settings_3dgs.connect(self.change_visualizer_settings_3dgs)
        self.visualizer_widget.signal_change_type.connect(self.visualizer_window.vis_type_changed)
        self.visualizer_widget.signal_get_current_view.connect(self.set_current_view)
        self.visualizer_widget.signal_pop_visualizer.connect(self.visualizer_window.on_embed_button_pressed)
        merger_widget.signal_merge_point_clouds.connect(self.io_controller.merge_point_clouds)
        rasterizer_tab.signal_rasterize.connect(self.rasterize_gaussians)

        tab_widget.addTab(input_tab, "I/O")
        tab_widget.addTab(self.transformation_picker, "Transformation")
        tab_widget.addTab(self.visualizer_widget, "Visualizer")
        tab_widget.addTab(rasterizer_tab, "Rasterizer")
        tab_widget.addTab(merger_widget, "Merging")

    def setup_registration_group(self, group_registration):
        group_registration.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        group_registration.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout()
        group_registration.setLayout(layout)
        group_registration.setTitle("Registration and evaluation")

        registration_tab = QTabWidget()

        local_registration_widget = LocalRegistrationTab()
        local_registration_widget.signal_do_registration.connect(
            self.registration_controller.execute_local_registration_normal)

        global_registration_widget = GlobalRegistrationTab()
        global_registration_widget.signal_do_ransac.connect(self.registration_controller.execute_ransac_registration_normal)
        global_registration_widget.signal_do_fgr.connect(self.registration_controller.execute_fgr_registration_normal)

        multi_scale_registration_widget = MultiScaleRegistrationTab()
        multi_scale_registration_widget.signal_do_registration.connect(
            self.registration_controller.execute_multiscale_registration)

        self.hem_widget = GaussianMixtureTab()
        self.hem_widget.signal_create_mixture.connect(self.downsampler_controller.create_mixture)
        self.hem_widget.signal_slider_changed.connect(self.active_pc_changed)
        self.downsampler_controller.signal_update_hem_slider.connect(self.hem_widget.update_hem_slider)

        evaluator_widget = EvaluationTab()
        evaluator_widget.signal_camera_change.connect(self.visualizer_window.apply_camera_view)
        evaluator_widget.signal_evaluate_registration.connect(self.registration_controller.evaluate_registration)

        plane_fitting_tab = PlaneFittingTab(self.data_repository)
        plane_fitting_tab.signal_fit_plane.connect(self.plane_fitting_controller.fit_plane)
        plane_fitting_tab.signal_clear_plane.connect(self.clear_planes)
        plane_fitting_tab.signal_merge_plane.connect(self.downsampler_controller.merge_plane_inliers)
        plane_fitting_tab.signal_error_message.connect(self.handle_error)
        plane_fitting_tab.signal_do_registration.connect(self.registration_controller.execute_local_registration_inlier)
        plane_fitting_tab.signal_do_fgr.connect(self.registration_controller.execute_fgr_registration_inlier)
        plane_fitting_tab.signal_do_ransac.connect(self.registration_controller.execute_ransac_registration_inlier)

        registration_tab.addTab(global_registration_widget, "Global")
        registration_tab.addTab(local_registration_widget, "Local")
        registration_tab.addTab(multi_scale_registration_widget, "Multiscale")
        registration_tab.addTab(self.hem_widget, "Mixture")
        registration_tab.addTab(plane_fitting_tab, "Plane fitting")
        registration_tab.addTab(evaluator_widget, "Evaluation")
        layout.addWidget(registration_tab)

    def register_controller_handlers(self):
        # Point Cloud IO Controller
        self.io_controller.signal_single_error.connect(self.handle_error)
        self.io_controller.signal_ui_update.connect(self.update_ui_after_pc_loaded)
        self.io_controller.load_point_clouds_signal.connect(self.visualizer_window.load_point_clouds)

        # Plane Fitting Controller
        self.plane_fitting_controller.signal_single_error.connect(self.handle_error)
        self.plane_fitting_controller.signal_list_error.connect(self.handle_error)

        # Registration Controller
        self.registration_controller.signal_single_error.connect(self.handle_error)
        self.registration_controller.signal_list_error.connect(self.handle_error)
        self.registration_controller.signal_success_message.connect(self.create_success_dialog)

    def register_repository_handlers(self):
        self.data_repository.signal_planes_changed.connect(self.visualizer_window.add_planes)
        self.ui_repository.signal_transformation_changed.connect(self.update_point_clouds)

    # region Event Handlers
    def update_point_clouds(self, transformation_matrix):
        dc1 = dc2 = None
        if self.visualizer_widget.get_use_debug_color():
            dc1, dc2 = self.visualizer_widget.get_debug_colors()

        self.visualizer_window.update_transform(transformation_matrix, dc1, dc2)

    def change_visualizer_settings_o3d(self, camera_view, dc1, dc2):
        self.visualizer_window.update_transform(self.ui_repository.transformation_matrix, dc1, dc2)
        self.visualizer_window.update_visualizer_settings_o3d(camera_view.zoom, camera_view.front, camera_view.lookat,
                                                              camera_view.up)

    def change_visualizer_settings_3dgs(self, camera_view, translate_speed, rotation_speed, roll_speed,
                                        background_color):
        self.visualizer_window.update_transform(self.ui_repository.transformation_matrix, None, None)
        self.visualizer_window.vis_3dgs.translate_speed = translate_speed
        self.visualizer_window.vis_3dgs.rotation_speed = rotation_speed
        self.visualizer_window.vis_3dgs.roll_speed = roll_speed
        self.visualizer_window.vis_3dgs.background_color = background_color
        self.visualizer_window.update_visualizer_settings_3dgs(camera_view.zoom, camera_view.front, camera_view.lookat,
                                                               camera_view.up)

    def set_current_view(self):
        self.visualizer_widget.set_visualizer_attributes(*self.visualizer_window.get_current_view())

    def rasterize_gaussians(self, width, height, scale, color, fx_supplied, fy_supplied):
        pc1 = self.data_repository.pc_gaussian_list_first[
            self.data_repository.current_index] if self.data_repository.pc_gaussian_list_first else None
        pc2 = self.data_repository.pc_gaussian_list_second[
            self.data_repository.current_index] if self.data_repository.pc_gaussian_list_second else None

        if not pc1 or not pc2:
            self.handle_error("Load two Gaussian point clouds for rasterization!")
            return

        if self.visualizer_window.is_ortho():
            self.handle_error("The current projection type is orthographical, which is invalid for rasterization.\n"
                              "Increase the FOV to continue!")
            return

        visualizer_camera = self.visualizer_window.get_camera
        fx, fy = fx_supplied, fy_supplied
        if fx == 0 and fy == 0:
            fx, fy = get_focal_from_intrinsics(self.visualizer_window.get_camera.intrinsics[0].numpy())

        new_camera = Camera(visualizer_camera.rotation.numpy(), visualizer_camera.position.numpy(),
                            fx, fy, "", width, height)

        progress_dialog = ProgressDialogFactory.get_progress_dialog("Loading", "Creating rasterized image...")
        worker = RasterizerWorker(pc1, pc2, self.ui_repository.transformation_matrix, new_camera, scale, color)

        thread = move_worker_to_thread(self, worker, self.create_raster_window,
                                       progress_handler=progress_dialog.setValue)
        thread.start()
        progress_dialog.exec()

    def create_raster_window(self, pix):
        self.raster_window = RasterImageViewer()
        self.raster_window.set_image(pix)
        self.raster_window.setWindowTitle("Rasterized point clouds")
        self.raster_window.setWindowModality(Qt.WindowModality.WindowModal)
        self.raster_window.show()

    def clear_planes(self):
        self.plane_fitting_controller.clear_planes()

        dc1, dc2 = None, None
        if self.visualizer_widget.get_use_debug_color():
            dc1, dc2 = self.visualizer_widget.get_debug_colors()

        self.visualizer_window.update_transform(self.ui_repository.transformation_matrix, dc1, dc2)

    def active_pc_changed(self, index):
        if self.data_repository.current_index == index:
            return

        self.data_repository.current_index = index

        dc1 = dc2 = None
        if self.visualizer_widget.get_use_debug_color():
            dc1, dc2 = self.visualizer_widget.get_debug_colors()

        params = PointCloudState(self.data_repository.pc_open3d_list_first[index],
                                 self.data_repository.pc_open3d_list_second[index],
                                 self.data_repository.pc_gaussian_list_first[index],
                                 self.data_repository.pc_gaussian_list_second[index],
                                 False,
                                 True,
                                 self.ui_repository.transformation_matrix,
                                 dc1,
                                 dc2)

        self.visualizer_window.load_point_clouds(params)

    def closeEvent(self, event):
        self.visualizer_window.vis_open3d.close()
        super(QMainWindow, self).closeEvent(event)

    def handle_error(self, error):
        if isinstance(error, list):
            self.create_error_list_dialog(error)
        else:
            self.create_error_dialog(str(error))

    def create_error_dialog(self, message):
        dialog = QErrorMessage(self)
        dialog.setModal(True)
        dialog.setWindowTitle("Error")
        dialog.showMessage(message)

    def create_error_list_dialog(self, error_list):
        message_dialog = QMessageBox(self)
        message_dialog.setModal(True)
        message_dialog.setWindowTitle("Error occurred")
        message_dialog.setText("The following error(s) occurred.\n Click \"Show details\" for more information!")
        message_dialog.setDetailedText("\n".join(error_list))
        message_dialog.exec()

    def create_success_dialog(self, title, message, detailed_text):
        message_dialog = QMessageBox(self)
        message_dialog.setModal(True)
        message_dialog.setWindowTitle(title)
        message_dialog.setText(message)
        if detailed_text != "":
            message_dialog.setDetailedText(detailed_text)
        message_dialog.exec()

    def update_ui_after_pc_loaded(self):
        self.hem_widget.set_slider_range(0)
        self.hem_widget.set_slider_enabled(False)
        self.transformation_picker.reset_transformation()
    # endregion
