from dataclasses import dataclass


@dataclass
class GaussianMixtureParams:
    hem_reduction: float = 3.0
    distance_delta: float = 3.0
    color_delta: float = 2.5
    decay_rate: float = 1.0
    cluster_level: int = 3
