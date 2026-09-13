import numpy as np
import pytest

from src.ransac import AffineRANSAC


def test_ransac_recovers_translation_with_no_outliers() -> None:
    """Test that RANSAC recovers a simple translation."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )

    scene_points = template_points + np.array(
        [100.0, 50.0],
        dtype=np.float64,
    )

    ransac = AffineRANSAC(
        iterations=100,
        reprojection_threshold=1.0,
        random_seed=42,
    )

    result = ransac.estimate(
        template_points,
        scene_points,
    )

    assert result.inlier_count == 5

    assert result.transform.a == pytest.approx(1.0)
    assert result.transform.b == pytest.approx(0.0)
    assert result.transform.tx == pytest.approx(100.0)

    assert result.transform.c == pytest.approx(0.0)
    assert result.transform.d == pytest.approx(1.0)
    assert result.transform.ty == pytest.approx(50.0)


def test_ransac_rejects_outliers() -> None:
    """Test that RANSAC rejects incorrect correspondences."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
            [20.0, 5.0],
            [5.0, 20.0],
            [30.0, 30.0],
        ],
        dtype=np.float64,
    )

    # True transformation:
    #
    # x_scene = x_template + 100
    # y_scene = y_template + 50

    scene_points = template_points + np.array(
        [100.0, 50.0],
        dtype=np.float64,
    )

    # Deliberately corrupt the last three correspondences.
    scene_points[5] = [500.0, 500.0]
    scene_points[6] = [700.0, 100.0]
    scene_points[7] = [50.0, 700.0]

    ransac = AffineRANSAC(
        iterations=500,
        reprojection_threshold=1.0,
        min_inliers=5,
        random_seed=42,
    )

    result = ransac.estimate(
        template_points,
        scene_points,
    )

    assert result.inlier_count == 5

    assert result.transform.a == pytest.approx(1.0)
    assert result.transform.b == pytest.approx(0.0)
    assert result.transform.tx == pytest.approx(100.0)

    assert result.transform.c == pytest.approx(0.0)
    assert result.transform.d == pytest.approx(1.0)
    assert result.transform.ty == pytest.approx(50.0)

    assert result.inlier_mask.tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_ransac_recovers_scaled_transform() -> None:
    """Test RANSAC with a non-uniform scaling transformation."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
            [20.0, 5.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.column_stack(
        (
            2.0 * template_points[:, 0] + 100.0,
            3.0 * template_points[:, 1] + 50.0,
        )
    )

    ransac = AffineRANSAC(
        iterations=200,
        reprojection_threshold=1.0,
        random_seed=42,
    )

    result = ransac.estimate(
        template_points,
        scene_points,
    )

    assert result.inlier_count == 6

    assert result.transform.a == pytest.approx(2.0)
    assert result.transform.b == pytest.approx(0.0)
    assert result.transform.tx == pytest.approx(100.0)

    assert result.transform.c == pytest.approx(0.0)
    assert result.transform.d == pytest.approx(3.0)
    assert result.transform.ty == pytest.approx(50.0)


def test_ransac_rejects_too_few_points() -> None:
    """Test that RANSAC requires at least three correspondences."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
        ],
        dtype=np.float64,
    )

    ransac = AffineRANSAC()

    with pytest.raises(
        ValueError,
        match="At least 3",
    ):
        ransac.estimate(
            template_points,
            scene_points,
        )


def test_ransac_rejects_mismatched_points() -> None:
    """Test that template and scene points must have equal length."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
        ],
        dtype=np.float64,
    )

    ransac = AffineRANSAC()

    with pytest.raises(
        ValueError,
        match="same number",
    ):
        ransac.estimate(
            template_points,
            scene_points,
        )


def test_ransac_detects_collinear_samples() -> None:
    """Test the collinearity check used by RANSAC."""

    collinear_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
        ],
        dtype=np.float64,
    )

    assert AffineRANSAC._are_collinear(collinear_points)

    non_collinear_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )

    assert not AffineRANSAC._are_collinear(non_collinear_points)
