import cv2
import numpy as np
import pytest

from src.matcher import FeatureMatch
from src.nms import Detection
from src.visualization import Visualizer


def test_draw_matches_returns_image() -> None:
    """Match visualization should return a valid image."""

    template_image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    scene_image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    template_keypoints = [
        cv2.KeyPoint(
            x=50.0,
            y=50.0,
            size=10.0,
            angle=0.0,
        )
    ]

    scene_keypoints = [
        cv2.KeyPoint(
            x=100.0,
            y=100.0,
            size=10.0,
            angle=0.0,
        )
    ]

    matches = [
        FeatureMatch(
            scene_index=0,
            template_index=0,
            distance=10.0,
        )
    ]

    result = Visualizer.draw_matches(
        template_image,
        scene_image,
        template_keypoints,
        scene_keypoints,
        matches,
    )

    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_draw_matches_has_expected_dimensions() -> None:
    """Match visualization should place images side by side."""

    template_image = np.zeros(
        (100, 150, 3),
        dtype=np.uint8,
    )

    scene_image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    template_keypoints = [
        cv2.KeyPoint(
            x=50.0,
            y=50.0,
            size=10.0,
            angle=0.0,
        )
    ]

    scene_keypoints = [
        cv2.KeyPoint(
            x=100.0,
            y=50.0,
            size=10.0,
            angle=0.0,
        )
    ]

    matches = [
        FeatureMatch(
            scene_index=0,
            template_index=0,
            distance=10.0,
        )
    ]

    result = Visualizer.draw_matches(
        template_image,
        scene_image,
        template_keypoints,
        scene_keypoints,
        matches,
    )

    assert result.shape[0] == 100
    assert result.shape[1] == 350


def test_draw_detections_returns_copy() -> None:
    """Drawing detections should not modify the original image."""

    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    original = image.copy()

    detections = [
        Detection(
            box=(50.0, 50.0, 150.0, 150.0),
            score=10.0,
        )
    ]

    result = Visualizer.draw_detections(
        image,
        detections,
    )

    assert np.array_equal(image, original)
    assert not np.array_equal(result, original)


def test_draw_oriented_detection() -> None:
    """Projected corners should be drawn as an oriented polygon."""

    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    detection = Detection(
        box=(40.0, 40.0, 160.0, 160.0),
        score=10.0,
        corners=(
            (100.0, 40.0),
            (160.0, 100.0),
            (100.0, 160.0),
            (40.0, 100.0),
        ),
    )

    result = Visualizer.draw_detections(image, [detection])

    assert not np.array_equal(result, image)
    assert np.any(result[40, 100] != 0)
    assert np.all(result[40, 40] == 0)


def test_draw_detections_preserves_dimensions() -> None:
    """Detection visualization should preserve image dimensions."""

    image = np.zeros(
        (300, 400, 3),
        dtype=np.uint8,
    )

    detections = [
        Detection(
            box=(50.0, 50.0, 150.0, 150.0),
            score=10.0,
        )
    ]

    result = Visualizer.draw_detections(
        image,
        detections,
    )

    assert result.shape == image.shape


def test_draw_detections_with_no_detections() -> None:
    """An image with no detections should remain unchanged."""

    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    result = Visualizer.draw_detections(
        image,
        [],
    )

    np.testing.assert_array_equal(
        result,
        image,
    )


def test_save_creates_output_file(tmp_path: pytest.TempPathFactory) -> None:
    """Visualizer.save should create the requested image file."""

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    output_path = tmp_path / "output.jpg"

    Visualizer.save(
        image,
        output_path,
    )

    assert output_path.exists()

    loaded = cv2.imread(
        str(output_path),
    )

    assert loaded is not None
    assert loaded.shape == image.shape


def test_save_creates_parent_directories(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Visualizer.save should create missing parent directories."""

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    output_path = tmp_path / "nested" / "directory" / "output.jpg"

    Visualizer.save(
        image,
        output_path,
    )

    assert output_path.exists()
