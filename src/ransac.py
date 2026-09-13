from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .affine import AffineEstimator, AffineTransform

PointArray = npt.NDArray[np.float64]


@dataclass
class RANSACResult:
    """Result of robust affine estimation."""

    transform: AffineTransform
    inlier_mask: npt.NDArray[np.bool_]
    error: float

    @property
    def inlier_count(self) -> int:
        """Return the number of inliers."""

        return int(np.count_nonzero(self.inlier_mask))


class AffineRANSAC:
    """Robustly estimate an affine transformation using RANSAC."""

    def __init__(
        self,
        iterations: int = 500,
        reprojection_threshold: float = 5.0,
        min_inliers: int = 3,
        random_seed: int | None = 42,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive.")

        if reprojection_threshold <= 0:
            raise ValueError("reprojection_threshold must be positive.")

        if min_inliers < 3:
            raise ValueError("min_inliers must be at least 3.")

        self.iterations = iterations
        self.reprojection_threshold = reprojection_threshold
        self.min_inliers = min_inliers

        self.rng = np.random.default_rng(random_seed)
        self.estimator = AffineEstimator()

    def estimate(
        self,
        template_points: PointArray,
        scene_points: PointArray,
    ) -> RANSACResult:
        """Estimate an affine transformation robustly."""

        self._validate_inputs(
            template_points,
            scene_points,
        )

        best_transform: AffineTransform | None = None
        best_inlier_mask: npt.NDArray[np.bool_] | None = None
        best_error = float("inf")
        best_inlier_count = 0

        point_count = len(template_points)

        for _ in range(self.iterations):
            sample_indices = self.rng.choice(
                point_count,
                size=3,
                replace=False,
            )

            sample_template = template_points[sample_indices]
            sample_scene = scene_points[sample_indices]

            if self._are_collinear(sample_template):
                continue

            try:
                transform = self.estimator.estimate(
                    sample_template,
                    sample_scene,
                )
            except ValueError:
                continue

            errors = self._reprojection_errors(
                transform,
                template_points,
                scene_points,
            )

            inlier_mask = errors <= self.reprojection_threshold

            inlier_count = int(np.count_nonzero(inlier_mask))

            if inlier_count < self.min_inliers:
                continue

            total_error = float(np.sum(errors[inlier_mask]))

            is_better = inlier_count > best_inlier_count or (
                inlier_count == best_inlier_count and total_error < best_error
            )

            if is_better:
                best_transform = transform
                best_inlier_mask = inlier_mask
                best_error = total_error
                best_inlier_count = inlier_count

        if best_transform is None or best_inlier_mask is None:
            raise RuntimeError("RANSAC failed to find a valid affine transformation.")

        refined_transform = self.estimator.estimate(
            template_points[best_inlier_mask],
            scene_points[best_inlier_mask],
        )

        refined_errors = self._reprojection_errors(
            refined_transform,
            template_points,
            scene_points,
        )

        refined_inlier_mask = refined_errors <= self.reprojection_threshold

        refined_error = float(np.sum(refined_errors[refined_inlier_mask]))

        if np.count_nonzero(refined_inlier_mask) < self.min_inliers:
            raise RuntimeError("Refined affine transformation has too few inliers.")

        return RANSACResult(
            transform=refined_transform,
            inlier_mask=refined_inlier_mask,
            error=refined_error,
        )

    @staticmethod
    def _reprojection_errors(
        transform: AffineTransform,
        template_points: PointArray,
        scene_points: PointArray,
    ) -> npt.NDArray[np.float64]:
        """Calculate Euclidean reprojection errors."""

        predicted_points = transform.transform(
            template_points,
        )

        differences = predicted_points - scene_points

        return np.linalg.norm(
            differences,
            axis=1,
        )

    @staticmethod
    def _are_collinear(
        points: PointArray,
    ) -> bool:
        """Check whether three points are approximately collinear."""

        if len(points) != 3:
            raise ValueError("Exactly three points are required.")

        p1, p2, p3 = points

        area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        return bool(np.isclose(area, 0.0))

    @staticmethod
    def _validate_inputs(
        template_points: PointArray,
        scene_points: PointArray,
    ) -> None:
        """Validate RANSAC input point correspondences."""

        if template_points.ndim != 2:
            raise ValueError("template_points must be a 2D array.")

        if scene_points.ndim != 2:
            raise ValueError("scene_points must be a 2D array.")

        if template_points.shape[1] != 2:
            raise ValueError("template_points must have shape (N, 2).")

        if scene_points.shape[1] != 2:
            raise ValueError("scene_points must have shape (N, 2).")

        if len(template_points) != len(scene_points):
            raise ValueError(
                "Template and scene points must have the same "
                "number of correspondences."
            )

        if len(template_points) < 3:
            raise ValueError("At least 3 correspondences are required.")
