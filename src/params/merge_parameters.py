from dataclasses import dataclass


@dataclass
class GaussianMixtureParams:
    hem_reduction: float
    distance_delta: float
    color_delta: float
    decay_rate: float
    cluster_level: int
