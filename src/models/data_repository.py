from PySide6.QtCore import QObject, Signal


class DataRepository(QObject):
    signal_planes_changed = Signal(list)

    def __init__(self):
        super().__init__()

        # Point cloud output of the 3D Gaussian Splatting
        self.pc_gaussian_list_first = []
        self.pc_gaussian_list_second = []

        # Open3D point clouds to display
        self.pc_open3d_list_first = []
        self.pc_open3d_list_second = []

        # Plane coefficients and indices
        self.first_plane_coefficients = []
        self.second_plane_coefficients = []
        self.first_plane_indices = []
        self.second_plane_indices = []
        # list of planes
        self._planes = []

        self.current_index = 0

        # Dataclass that stores the results and parameters of the last local registration
        self.local_registration_data = None

    @property
    def planes(self):
        return self._planes

    @planes.setter
    def planes(self, planes):
        self._planes = planes
        self.signal_planes_changed.emit(planes)
