import numpy as np
import pytest

from src.affine import AffineTransform
from src.detection import DetectionProjector


def test_identity_projection() -> None:
    """Template corners should remain unchanged under identity."""

    transform = AffineTransform(
        a=1.0,
        b=0.0,
        tx=0.0,
        c=0.0,
        d=1.0,
        ty=0.0,
    )

    projector = DetectionProjector()

    result = projector.project(
        transform=transform,
        template_width=100,
        template_height=50,
    )

    expected_corners = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 50.0],
            [0.0, 50.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.corners,
        expected_corners,
    )

    assert result.bounding_box == pytest.approx((0.0, 0.0, 100.0, 50.0))


def test_translation_projection() -> None:
    """Template corners should be translated correctly."""

    transform = AffineTransform(
        a=1.0,
        b=0.0,
        tx=100.0,
        c=0.0,
        d=1.0,
        ty=50.0,
    )

    projector = DetectionProjector()

    result = projector.project(
        transform=transform,
        template_width=100,
        template_height=50,
    )

    expected_corners = np.array(
        [
            [100.0, 50.0],
            [200.0, 50.0],
            [200.0, 100.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.corners,
        expected_corners,
    )

    assert result.bounding_box == pytest.approx((100.0, 50.0, 200.0, 100.0))


def test_scaling_projection() -> None:
    """Template corners should be scaled correctly."""

    transform = AffineTransform(
        a=2.0,
        b=0.0,
        tx=0.0,
        c=0.0,
        d=3.0,
        ty=0.0,
    )

    projector = DetectionProjector()

    result = projector.project(
        transform=transform,
        template_width=100,
        template_height=50,
    )

    expected_corners = np.array(
        [
            [0.0, 0.0],
            [200.0, 0.0],
            [200.0, 150.0],
            [0.0, 150.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.corners,
        expected_corners,
    )

    assert result.bounding_box == pytest.approx((0.0, 0.0, 200.0, 150.0))


def test_rotation_projection() -> None:
    """Template corners should rotate correctly."""

    transform = AffineTransform(
        a=0.0,
        b=-1.0,
        tx=0.0,
        c=1.0,
        d=0.0,
        ty=0.0,
    )

    projector = DetectionProjector()

    result = projector.project(
        transform=transform,
        template_width=100,
        template_height=50,
    )

    expected_corners = np.array(
        [
            [0.0, 0.0],
            [0.0, 100.0],
            [-50.0, 100.0],
            [-50.0, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.corners,
        expected_corners,
    )

    assert result.bounding_box == pytest.approx((-50.0, 0.0, 0.0, 100.0))


def test_combined_transform_projection() -> None:
    """Test rotation, scaling, and translation together."""

    transform = AffineTransform(
        a=2.0,
        b=0.0,
        tx=100.0,
        c=0.0,
        d=2.0,
        ty=50.0,
    )

    projector = DetectionProjector()

    result = projector.project(
        transform=transform,
        template_width=50,
        template_height=25,
    )

    expected_corners = np.array(
        [
            [100.0, 50.0],
            [200.0, 50.0],
            [200.0, 100.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.corners,
        expected_corners,
    )

    assert result.bounding_box == pytest.approx((100.0, 50.0, 200.0, 100.0))


def test_to_detection() -> None:
    """Test conversion from projected template to Detection."""

    transform = AffineTransform(
        a=1.0,
        b=0.0,
        tx=100.0,
        c=0.0,
        d=1.0,
        ty=50.0,
    )

    projector = DetectionProjector()

    projected = projector.project(
        transform=transform,
        template_width=100,
        template_height=50,
    )

    detection = projector.to_detection(
        projected=projected,
        score=12.5,
    )

    assert detection.box == pytest.approx((100.0, 50.0, 200.0, 100.0))
    assert detection.corners == (
        (100.0, 50.0),
        (200.0, 50.0),
        (200.0, 100.0),
        (100.0, 100.0),
    )
    assert detection.score == pytest.approx(12.5)


def test_invalid_template_width() -> None:
    """Template width must be positive."""

    transform = AffineTransform(
        a=1.0,
        b=0.0,
        tx=0.0,
        c=0.0,
        d=1.0,
        ty=0.0,
    )

    projector = DetectionProjector()

    with pytest.raises(
        ValueError,
        match="template_width must be positive",
    ):
        projector.project(
            transform=transform,
            template_width=0,
            template_height=50,
        )


def test_invalid_template_height() -> None:
    """Template height must be positive."""

    transform = AffineTransform(
        a=1.0,
        b=0.0,
        tx=0.0,
        c=0.0,
        d=1.0,
        ty=0.0,
    )

    projector = DetectionProjector()

    with pytest.raises(
        ValueError,
        match="template_height must be positive",
    ):
        projector.project(
            transform=transform,
            template_width=100,
            template_height=-10,
        )
