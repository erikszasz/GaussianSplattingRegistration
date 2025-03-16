from enum import Enum

import open3d as o3d


class GlobalRegistrationType(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name):
        self.instance_name = name

    RANSAC = "RANSAC"
    FGR = "FGR"


class RANSACEstimationMethod(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name):
        self.instance_name = name

    TransformationEstimationPointToPoint = "Point-To-Point"
    TransformationEstimationPointToPlane = "Point-To-Plane"
    TransformationEstimationForGeneralizedICP = "For GICP"
    TransformationEstimationForColoredICP = "For CICP"


def get_estimation_method_from_enum(estimation_method):
    match estimation_method:
        case RANSACEstimationMethod.TransformationEstimationPointToPoint:
            return o3d.pipelines.registration.TransformationEstimationPointToPoint()
        case RANSACEstimationMethod.TransformationEstimationPointToPlane:
            return o3d.pipelines.registration.TransformationEstimationPointToPlane()
        case RANSACEstimationMethod.TransformationEstimationForGeneralizedICP:
            return o3d.pipelines.registration.TransformationEstimationForColoredICP()
        case RANSACEstimationMethod.TransformationEstimationForColoredICP:
            return o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()


def do_ransac_registration(point_cloud_first, point_cloud_second, params):
    source_down, source_fpfh = preprocess_point_cloud(point_cloud_first, params.voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(point_cloud_second, params.voxel_size)
    real_estimation_method = get_estimation_method_from_enum(params.estimation_method)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, params.mutual_filter,
        params.max_correspondence,
        real_estimation_method,
        params.ransac_n,
        params.checkers,
        o3d.pipelines.registration.RANSACConvergenceCriteria(params.max_iteration, params.confidence))

    return result


def do_fgr_registration(point_cloud_first, point_cloud_second, registration_params):
    source_down, source_fpfh = preprocess_point_cloud(point_cloud_first, registration_params.voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(point_cloud_second, registration_params.voxel_size)

    options = o3d.pipelines.registration.FastGlobalRegistrationOption(registration_params.division_factor,
                                                                      registration_params.use_absolute_scale,
                                                                      registration_params.decrease_mu,
                                                                      registration_params.maximum_correspondence,
                                                                      registration_params.max_iterations,
                                                                      registration_params.tuple_scale,
                                                                      registration_params.max_tuple_count,
                                                                      registration_params.tuple_test)

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source_down, target_down,
                                                                                   source_fpfh,
                                                                                   target_fpfh, options)

    return result


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
