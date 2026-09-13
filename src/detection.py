from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from .affine import AffineTransform
from .nms import Detection

PointArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ProjectedTemplate:
    """Template corners projected into the scene."""

    corners: PointArray
    bounding_box: tuple[float, float, float, float]


class DetectionProjector:
    """Project template geometry into the scene."""

    def project(
        self,
        transform: AffineTransform,
        template_width: int,
        template_height: int,
    ) -> ProjectedTemplate:
        """Project the four template corners using an affine transform."""

        if template_width <= 0:
            raise ValueError("template_width must be positive.")

        if template_height <= 0:
            raise ValueError("template_height must be positive.")

        corners = np.array(
            [
                [0.0, 0.0],
                [float(template_width), 0.0],
                [float(template_width), float(template_height)],
                [0.0, float(template_height)],
            ],
            dtype=np.float64,
        )

        projected_corners = transform.transform(
            corners,
        )

        bounding_box = self._bounding_box(
            projected_corners,
        )

        return ProjectedTemplate(
            corners=projected_corners,
            bounding_box=bounding_box,
        )

    @staticmethod
    def _bounding_box(
        points: PointArray,
    ) -> tuple[float, float, float, float]:
        """Calculate an axis-aligned bounding box around points."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2).")

        if len(points) == 0:
            raise ValueError("points cannot be empty.")

        x_min = float(np.min(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        x_max = float(np.max(points[:, 0]))
        y_max = float(np.max(points[:, 1]))

        return (
            x_min,
            y_min,
            x_max,
            y_max,
        )

    @staticmethod
    def to_detection(
        projected: ProjectedTemplate,
        score: float,
    ) -> Detection:
        """Convert a projected template into a detection."""

        corners: tuple[tuple[float, float], ...] = tuple(
            (
                cast(float, projected.corners[index, 0]),
                cast(float, projected.corners[index, 1]),
            )
            for index in range(len(projected.corners))
        )

        return Detection(
            box=projected.bounding_box,
            score=score,
            corners=corners,
        )
