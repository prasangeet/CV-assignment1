from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DescriptorArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class FeatureMatch:
    """A correspondence between a scene and template descriptor."""

    scene_index: int
    template_index: int
    distance: float


class DescriptorMatcher:
    """Match scene descriptors against template descriptors."""

    def __init__(self, ratio_threshold: float = 0.75) -> None:
        if not 0.0 < ratio_threshold < 1.0:
            raise ValueError("ratio_threshold must be between 0 and 1.")

        self.ratio_threshold = ratio_threshold

    def match(
        self,
        scene_descriptors: DescriptorArray,
        template_descriptors: DescriptorArray,
    ) -> list[FeatureMatch]:
        """Find valid scene-to-template matches using Lowe's ratio test."""

        if scene_descriptors.ndim != 2:
            raise ValueError("Scene descriptors must be a 2D array.")

        if template_descriptors.ndim != 2:
            raise ValueError("Template descriptors must be a 2D array.")

        if scene_descriptors.shape[1] != template_descriptors.shape[1]:
            raise ValueError(
                "Scene and template descriptors must have the same dimension."
            )

        matches: list[FeatureMatch] = []

        for scene_index, scene_descriptor in enumerate(scene_descriptors):
            distances = np.linalg.norm(
                template_descriptors - scene_descriptor,
                axis=1,
            )

            if len(distances) < 2:
                continue

            nearest_indices = np.argsort(distances)[:2]

            best_index = int(nearest_indices[0])
            second_best_index = int(nearest_indices[1])

            best_distance = float(distances[best_index])
            second_best_distance = float(distances[second_best_index])

            if second_best_distance == 0.0:
                continue

            ratio = best_distance / second_best_distance

            if ratio < self.ratio_threshold:
                matches.append(
                    FeatureMatch(
                        scene_index=scene_index,
                        template_index=best_index,
                        distance=best_distance,
                    )
                )

        return matches
