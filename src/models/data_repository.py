
class DataRepository:
    def __init__(self):
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

        self.current_index = 0

        # Dataclass that stores the results and parameters of the last local registration
        self.local_registration_data = None