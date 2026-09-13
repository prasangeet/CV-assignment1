from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

PointArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AffineTransform:
    """A 2D affine transformation."""

    a: float
    b: float
    tx: float
    c: float
    d: float
    ty: float

    def transform(
        self,
        points: PointArray,
    ) -> PointArray:
        """Transform a collection of 2D points."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2).")

        x = points[:, 0]
        y = points[:, 1]

        transformed_x = self.a * x + self.b * y + self.tx

        transformed_y = self.c * x + self.d * y + self.ty

        return np.column_stack((transformed_x, transformed_y))

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Return the 3x3 homogeneous transformation matrix."""

        return np.array(
            [
                [self.a, self.b, self.tx],
                [self.c, self.d, self.ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def determinant(self) -> float:
        """Return the determinant of the linear component."""

        return self.a * self.d - self.b * self.c


class AffineEstimator:
    """Estimate a 2D affine transformation using least squares."""

    def estimate(
        self,
        template_points: PointArray,
        scene_points: PointArray,
    ) -> AffineTransform:
        """Estimate an affine transformation from point correspondences."""

        self._validate_points(
            template_points,
            "template_points",
        )

        self._validate_points(
            scene_points,
            "scene_points",
        )

        if len(template_points) != len(scene_points):
            raise ValueError(
                "Template and scene points must have the same number of correspondences."
            )

        if len(template_points) < 3:
            raise ValueError("At least 3 point correspondences are required.")

        A = self._build_system(
            template_points,
        )

        b = scene_points.reshape(-1)

        coefficients, _, rank, _ = np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )

        if rank < 6:
            raise ValueError(
                "Point correspondences are degenerate; the affine transformation cannot be uniquely estimated."
            )

        return AffineTransform(
            a=float(coefficients[0]),
            b=float(coefficients[1]),
            tx=float(coefficients[2]),
            c=float(coefficients[3]),
            d=float(coefficients[4]),
            ty=float(coefficients[5]),
        )

    @staticmethod
    def _build_system(
        template_points: PointArray,
    ) -> PointArray:
        """Build the linear system A for affine estimation."""

        n = len(template_points)

        A = np.zeros(
            (2 * n, 6),
            dtype=np.float64,
        )

        for i, (x, y) in enumerate(template_points):
            A[2 * i] = [
                x,
                y,
                1.0,
                0.0,
                0.0,
                0.0,
            ]

            A[2 * i + 1] = [
                0.0,
                0.0,
                0.0,
                x,
                y,
                1.0,
            ]

        return A

    @staticmethod
    def _validate_points(
        points: PointArray,
        name: str,
    ) -> None:
        """Validate a point array."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2).")

        if len(points) == 0:
            raise ValueError(f"{name} cannot be empty.")
