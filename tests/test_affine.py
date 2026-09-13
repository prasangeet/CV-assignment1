import numpy as np
import pytest

from src.affine import AffineEstimator, AffineTransform


def test_affine_identity_transform() -> None:
    """Test estimation of an identity transformation."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    scene_points = template_points.copy()

    estimator = AffineEstimator()

    transform = estimator.estimate(
        template_points,
        scene_points,
    )

    assert transform.a == pytest.approx(1.0)
    assert transform.b == pytest.approx(0.0)
    assert transform.tx == pytest.approx(0.0)

    assert transform.c == pytest.approx(0.0)
    assert transform.d == pytest.approx(1.0)
    assert transform.ty == pytest.approx(0.0)


def test_affine_translation() -> None:
    """Test estimation of a pure translation."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [100.0, 50.0],
            [110.0, 50.0],
            [100.0, 60.0],
            [110.0, 60.0],
        ],
        dtype=np.float64,
    )

    estimator = AffineEstimator()

    transform = estimator.estimate(
        template_points,
        scene_points,
    )

    assert transform.a == pytest.approx(1.0)
    assert transform.b == pytest.approx(0.0)
    assert transform.tx == pytest.approx(100.0)

    assert transform.c == pytest.approx(0.0)
    assert transform.d == pytest.approx(1.0)
    assert transform.ty == pytest.approx(50.0)


def test_affine_scaling() -> None:
    """Test estimation of a scaling transformation."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [0.0, 0.0],
            [20.0, 0.0],
            [0.0, 30.0],
            [20.0, 30.0],
        ],
        dtype=np.float64,
    )

    estimator = AffineEstimator()

    transform = estimator.estimate(
        template_points,
        scene_points,
    )

    assert transform.a == pytest.approx(2.0)
    assert transform.b == pytest.approx(0.0)
    assert transform.tx == pytest.approx(0.0)

    assert transform.c == pytest.approx(0.0)
    assert transform.d == pytest.approx(3.0)
    assert transform.ty == pytest.approx(0.0)


def test_affine_rotation() -> None:
    """Test estimation of a 90-degree rotation."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [-10.0, 0.0],
            [-10.0, 10.0],
        ],
        dtype=np.float64,
    )

    estimator = AffineEstimator()

    transform = estimator.estimate(
        template_points,
        scene_points,
    )

    assert transform.a == pytest.approx(0.0)
    assert transform.b == pytest.approx(-1.0)
    assert transform.tx == pytest.approx(0.0)

    assert transform.c == pytest.approx(1.0)
    assert transform.d == pytest.approx(0.0)
    assert transform.ty == pytest.approx(0.0)


def test_affine_transform_points() -> None:
    """Test applying an affine transformation to points."""

    transform = AffineTransform(
        a=2.0,
        b=0.0,
        tx=10.0,
        c=0.0,
        d=3.0,
        ty=20.0,
    )

    points = np.array(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [10.0, 20.0],
        ],
        dtype=np.float64,
    )

    transformed = transform.transform(points)

    expected = np.array(
        [
            [10.0, 20.0],
            [20.0, 50.0],
            [30.0, 80.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        transformed,
        expected,
    )


def test_affine_matrix() -> None:
    """Test the homogeneous affine matrix."""

    transform = AffineTransform(
        a=2.0,
        b=3.0,
        tx=10.0,
        c=4.0,
        d=5.0,
        ty=20.0,
    )

    expected = np.array(
        [
            [2.0, 3.0, 10.0],
            [4.0, 5.0, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        transform.matrix,
        expected,
    )


def test_affine_determinant() -> None:
    """Test the determinant of the linear component."""

    transform = AffineTransform(
        a=2.0,
        b=3.0,
        tx=10.0,
        c=4.0,
        d=5.0,
        ty=20.0,
    )

    # det = ad - bc
    #     = (2)(5) - (3)(4)
    #     = -2

    assert transform.determinant == pytest.approx(-2.0)


def test_affine_requires_at_least_three_points() -> None:
    """Test that fewer than three correspondences are rejected."""

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

    estimator = AffineEstimator()

    with pytest.raises(ValueError, match="At least 3"):
        estimator.estimate(
            template_points,
            scene_points,
        )


def test_affine_rejects_mismatched_correspondences() -> None:
    """Test that both point sets must have equal length."""

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

    estimator = AffineEstimator()

    with pytest.raises(
        ValueError,
        match="same number of correspondences",
    ):
        estimator.estimate(
            template_points,
            scene_points,
        )


def test_affine_rejects_invalid_point_shape() -> None:
    """Test that points must have shape (N, 2)."""

    template_points = np.array(
        [0.0, 10.0, 20.0],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
            [120.0, 100.0],
        ],
        dtype=np.float64,
    )

    estimator = AffineEstimator()

    with pytest.raises(
        ValueError,
        match="template_points must have shape",
    ):
        estimator.estimate(
            template_points,
            scene_points,
        )


def test_affine_rejects_degenerate_points() -> None:
    """Test that collinear points cannot define a unique affine transform."""

    template_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float64,
    )

    scene_points = np.array(
        [
            [100.0, 100.0],
            [120.0, 100.0],
            [140.0, 100.0],
        ],
        dtype=np.float64,
    )

    estimator = AffineEstimator()

    with pytest.raises(
        ValueError,
        match="degenerate",
    ):
        estimator.estimate(
            template_points,
            scene_points,
        )
