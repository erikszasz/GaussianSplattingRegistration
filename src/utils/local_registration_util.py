from enum import Enum

import open3d as o3d


class KernelLossFunctionType(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name):
        self.instance_name = name

    Loss_None = "None"
    Tukey_Loss = "Tukey loss"
    Cauchy_Loss = "Cauchy loss"
    GMLoss = "GM loss"
    Huber_Loss = "Huber loss"


class LocalRegistrationType(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, name):
        self.instance_name = name

    ICP_Point_To_Point = "Point-to-Point ICP"
    ICP_Point_To_Plane = "Point-to-Plane ICP"
    ICP_Color = "Colored ICP"
    ICP_General = "Generalized ICP"


def get_estimation(registration_type, loss_function):
    if loss_function is None:
        return o3d.pipelines.registration.TransformationEstimationPointToPoint()

    match registration_type:
        case LocalRegistrationType.ICP_Point_To_Point:
            return o3d.pipelines.registration.TransformationEstimationPointToPoint()
        case LocalRegistrationType.ICP_Point_To_Plane:
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(loss_function)
        case LocalRegistrationType.ICP_Color:
            return o3d.pipelines.registration.TransformationEstimationForColoredICP(loss_function)
        case LocalRegistrationType.ICP_General:
            return o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss_function)


def get_convergence_criteria(relative_fitness, relative_rmse, max_iteration):
    return o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness, relative_rmse, max_iteration)


def get_rejection_loss(rejection_type, k_value, registration_type):
    if registration_type is LocalRegistrationType.ICP_Point_To_Point:
        return None

    if rejection_type is KernelLossFunctionType.Loss_None or k_value == 0.0:
        return o3d.pipelines.registration.L2Loss()

    match rejection_type:
        case KernelLossFunctionType.Tukey_Loss:
            return o3d.pipelines.registration.TukeyLoss(k_value)
        case KernelLossFunctionType.Cauchy_Loss:
            return o3d.pipelines.registration.CauchyLoss(k_value)
        case KernelLossFunctionType.GMLoss:
            return o3d.pipelines.registration.GMLoss(k_value)
        case KernelLossFunctionType.Huber_Loss:
            return o3d.pipelines.registration.HuberLoss(k_value)


def do_icp_registration(point_cloud_first, point_cloud_second, init_transform,
                        registration_params):
    loss_function = get_rejection_loss(registration_params.rejection_type, registration_params.k_value,
                                       registration_params.registration_type)
    estimation_method = get_estimation(registration_params.registration_type, loss_function)
    convergence_criteria = get_convergence_criteria(registration_params.relative_fitness,
                                                    registration_params.relative_rmse,
                                                    registration_params.max_iteration)

    max_correspondence = registration_params.max_correspondence
    match registration_params.registration_type:
        case LocalRegistrationType.ICP_Point_To_Point | LocalRegistrationType.ICP_Point_To_Plane:
            return o3d.pipelines.registration.registration_icp(point_cloud_first, point_cloud_second,
                                                               max_correspondence, init_transform,
                                                               estimation_method, convergence_criteria)
        case LocalRegistrationType.ICP_Color:
            return o3d.pipelines.registration.registration_colored_icp(point_cloud_first, point_cloud_second,
                                                                       max_correspondence, init_transform,
                                                                       estimation_method, convergence_criteria)
        case LocalRegistrationType.ICP_General:
            return o3d.pipelines.registration.registration_generalized_icp(point_cloud_first, point_cloud_second,
                                                                           max_correspondence, init_transform,
                                                                           estimation_method, convergence_criteria)
        case _:
            return None
