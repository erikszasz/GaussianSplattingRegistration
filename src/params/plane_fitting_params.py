from dataclasses import dataclass


@dataclass
class PlaneFittingParams:
    plane_count: int = 3
    iteration: int = 1000
    distance_threshold: float = 0.01
    normal_threshold: float = 0.9
    min_distance: float = 0.05
