import pytest

from src.nms import Detection, NonMaximumSuppression


def test_iou_identical_boxes() -> None:
    """Identical boxes should have IoU of 1."""

    box = (10.0, 10.0, 100.0, 100.0)

    iou = NonMaximumSuppression.iou(box, box)

    assert iou == pytest.approx(1.0)


def test_iou_non_overlapping_boxes() -> None:
    """Non-overlapping boxes should have IoU of 0."""

    box_a = (0.0, 0.0, 50.0, 50.0)
    box_b = (100.0, 100.0, 150.0, 150.0)

    iou = NonMaximumSuppression.iou(box_a, box_b)

    assert iou == pytest.approx(0.0)


def test_iou_partial_overlap() -> None:
    """Test IoU for partially overlapping boxes."""

    box_a = (0.0, 0.0, 100.0, 100.0)
    box_b = (50.0, 50.0, 150.0, 150.0)

    iou = NonMaximumSuppression.iou(box_a, box_b)

    # Intersection:
    # 50 x 50 = 2500
    #
    # Area A:
    # 100 x 100 = 10000
    #
    # Area B:
    # 100 x 100 = 10000
    #
    # Union:
    # 10000 + 10000 - 2500 = 17500
    #
    # IoU = 2500 / 17500

    assert iou == pytest.approx(2500.0 / 17500.0)


def test_iou_is_symmetric() -> None:
    """IoU should be identical regardless of box order."""

    box_a = (0.0, 0.0, 100.0, 100.0)
    box_b = (50.0, 50.0, 150.0, 150.0)

    iou_ab = NonMaximumSuppression.iou(
        box_a,
        box_b,
    )

    iou_ba = NonMaximumSuppression.iou(
        box_b,
        box_a,
    )

    assert iou_ab == pytest.approx(iou_ba)


def test_nms_keeps_highest_scoring_detection() -> None:
    """NMS should keep the highest-scoring overlapping detection."""

    detections = [
        Detection(
            box=(0.0, 0.0, 100.0, 100.0),
            score=0.9,
        ),
        Detection(
            box=(10.0, 10.0, 110.0, 110.0),
            score=0.7,
        ),
    ]

    nms = NonMaximumSuppression(
        iou_threshold=0.5,
    )

    result = nms.suppress(detections)

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.9)


def test_nms_keeps_non_overlapping_detections() -> None:
    """NMS should preserve detections that do not overlap."""

    detections = [
        Detection(
            box=(0.0, 0.0, 100.0, 100.0),
            score=0.9,
        ),
        Detection(
            box=(200.0, 200.0, 300.0, 300.0),
            score=0.8,
        ),
    ]

    nms = NonMaximumSuppression(
        iou_threshold=0.5,
    )

    result = nms.suppress(detections)

    assert len(result) == 2


def test_nms_processes_detections_by_score() -> None:
    """NMS should consider the highest-scoring detection first."""

    detections = [
        Detection(
            box=(10.0, 10.0, 110.0, 110.0),
            score=0.5,
        ),
        Detection(
            box=(0.0, 0.0, 100.0, 100.0),
            score=0.9,
        ),
        Detection(
            box=(300.0, 300.0, 400.0, 400.0),
            score=0.7,
        ),
    ]

    nms = NonMaximumSuppression(
        iou_threshold=0.5,
    )

    result = nms.suppress(detections)

    assert len(result) == 2

    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.7)


def test_nms_empty_input() -> None:
    """NMS should return an empty list for empty input."""

    nms = NonMaximumSuppression()

    result = nms.suppress([])

    assert result == []


def test_nms_rejects_invalid_threshold() -> None:
    """IoU threshold must be between 0 and 1."""

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        NonMaximumSuppression(
            iou_threshold=1.5,
        )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        NonMaximumSuppression(
            iou_threshold=-0.1,
        )
