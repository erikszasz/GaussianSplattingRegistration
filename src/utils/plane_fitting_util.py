import open3d as o3d
import numpy as np
import torch


def fit_planes(point_cloud: o3d.geometry.PointCloud, plane_count, iterations, threshold, normal_threshold,
               min_sample_distance):
    points_tensor = torch.from_numpy(np.array(point_cloud.points, dtype=np.float32))
    normal_tensor = torch.from_numpy(np.array(point_cloud.normals, dtype=np.float32))
    original_indices = torch.arange(points_tensor.shape[0])  # Track original indices
    plane_coefficients = []
    inlier_indices_list = []

    for _ in range(plane_count):
        best_plane, best_inliers = _fit_single_plane(points_tensor, normal_tensor, iterations, threshold,
                                                     normal_threshold, min_sample_distance)
        if best_plane is not None:
            plane_coefficients.append(best_plane)
            inlier_indices_list.append(original_indices[best_inliers])

            mask = torch.ones(points_tensor.shape[0], dtype=torch.bool)
            mask[best_inliers] = False

            points_tensor = points_tensor[mask]
            original_indices = original_indices[mask]
        else:
            break

        if points_tensor.shape[0] == 0:
            break

    return plane_coefficients, inlier_indices_list


def _fit_single_plane(points_tensor, normal_tensors, iterations, distance_threshold, normal_threshold,
                      min_sample_distance):
    best_plane = None
    max_inliers_count = 0
    best_inliers = None

    for _ in range(iterations):
        sampled_indices = sample_random_points(points_tensor, min_sample_distance)
        sampled_points = points_tensor[sampled_indices]

        p1, p2, p3 = sampled_points[0], sampled_points[1], sampled_points[2]
        normal = torch.cross(p2 - p1, p3 - p1, dim=0)
        normal /= normal.norm()
        d = -torch.dot(normal, p1)

        plane = np.hstack((normal.numpy(), d.numpy()))

        _, distances = project_point_onto_plane(points_tensor, plane)

        distance_inliers = (torch.abs(distances) < distance_threshold).nonzero(as_tuple=True)[0]

        if distance_inliers.size(0) > 0:
            aligned_normals = normal_tensors[distance_inliers]
            alignment = torch.abs(torch.matmul(aligned_normals, normal))

            normal_inliers = distance_inliers[alignment > normal_threshold]
            inliers_count = normal_inliers.size(0)

            if inliers_count > max_inliers_count:
                max_inliers_count = inliers_count
                best_plane = plane
                best_inliers = normal_inliers.numpy()

    return best_plane, best_inliers


def sample_random_points(points: torch.Tensor, min_distance: float) -> torch.Tensor:
    num_points = points.shape[0]

    def is_distant_enough(selected_points, candidate):
        distances = torch.norm(selected_points - candidate, dim=1)
        return torch.all(distances >= min_distance)

    selected_indices = []
    first_index = torch.randint(0, num_points, (1,)).item()
    selected_indices.append(first_index)

    while len(selected_indices) < 3:
        candidate_index = torch.randint(0, num_points, (1,)).item()
        candidate_point = points[candidate_index]
        selected_points = points[selected_indices]

        if is_distant_enough(selected_points, candidate_point):
            selected_indices.append(candidate_index)

    return torch.tensor(selected_indices)


def project_point_onto_plane(points, plane):
    a, b, c, d = plane
    normal = torch.tensor([a, b, c], dtype=torch.float32)
    normal = normal / normal.norm()

    distances = (torch.mm(points, normal.unsqueeze(1)).squeeze() + d) / normal.norm()

    projected_points = points - distances.unsqueeze(1) * normal
    return projected_points, distances


# TODO: Refactor this
def get_o3d_plane(plane_coefficients, points_tensor, color, resolution=10):
    a, b, c, d = plane_coefficients

    # Project points onto the plane
    normal = torch.tensor([a, b, c], dtype=torch.float32)
    normal /= torch.norm(normal)
    distances = torch.matmul(points_tensor, normal) + d
    projected_points = points_tensor - distances.unsqueeze(1) * normal

    # Compute the 2D bounding box in the plane's coordinate system
    u = torch.tensor([-b, a, 0], dtype=torch.float32)
    if torch.norm(u) == 0:
        u = torch.tensor([1, 0, 0], dtype=torch.float32)  # Handle edge case if normal is along z-axis
    u /= torch.norm(u)
    v = torch.linalg.cross(normal, u)

    # Transform points to the plane's local coordinate system
    plane_coords = torch.stack((torch.matmul(projected_points, u), torch.matmul(projected_points, v)), dim=-1)
    min_coords = plane_coords.min(dim=0).values
    max_coords = plane_coords.max(dim=0).values

    # Generate a grid of points in the local 2D plane coordinates
    x = torch.linspace(min_coords[0], max_coords[0], resolution)
    y = torch.linspace(min_coords[1], max_coords[1], resolution)
    x, y = torch.meshgrid(x, y, indexing="xy")

    # Convert the local 2D coordinates back to 3D coordinates in the original space
    plane_points_3d = (x.unsqueeze(-1) * u + y.unsqueeze(-1) * v).reshape(-1, 3)
    plane_points_3d += normal.unsqueeze(0) * (-d / normal.norm())  # Add the plane's offset

    # Create the mesh
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_points_3d.numpy())

    # Define triangles for the grid
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            triangles.append([idx, idx + resolution, idx + 1])
            triangles.append([idx + resolution, idx + resolution + 1, idx + 1])

    # Duplicate triangles with reversed normals for double-sided visibility
    reversed_triangles = [[tri[2], tri[1], tri[0]] for tri in triangles]
    triangles.extend(reversed_triangles)

    plane_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    plane_mesh.paint_uniform_color(color)

    return plane_mesh
