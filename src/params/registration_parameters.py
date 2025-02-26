from dataclasses import dataclass, field

from utils.global_registration_util import RANSACEstimationMethod
from utils.local_registration_util import LocalRegistrationType, KernelLossFunctionType


@dataclass
class LocalRegistrationParams:
    registration_type: LocalRegistrationType = LocalRegistrationType.ICP_Point_To_Point
    max_correspondence: float = 5.0
    relative_fitness: float = 0.000001
    relative_rmse: float = 0.000001
    max_iteration: int = 30
    rejection_type: KernelLossFunctionType = KernelLossFunctionType.Loss_None
    k_value: float = 0.0


@dataclass
class FGRRegistrationParams:
    voxel_size: float = 0.05
    division_factor: float = 1.4
    use_absolute_scale: bool = False
    decrease_mu: bool = False
    maximum_correspondence: float = 0.025
    max_iterations: int = 64
    tuple_scale: float = 0.95
    max_tuple_count: int = 1000
    tuple_test: bool = True


@dataclass
class RANSACRegistrationParams:
    voxel_size: float = 0.05
    mutual_filter: bool = False
    max_correspondence: float = 5.0
    estimation_method: RANSACEstimationMethod = RANSACEstimationMethod.TransformationEstimationPointToPoint
    ransac_n: int = 3
    checkers: list = field(default_factory=list)
    max_iteration: int = 100000
    confidence: float = 0.999
