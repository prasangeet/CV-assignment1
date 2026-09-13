import cv2
import pytest

from src.hough import GeneralizedHough, HoughVote
from src.matcher import FeatureMatch


def create_keypoint(
    x: float,
    y: float,
    size: float,
    angle: float,
) -> cv2.KeyPoint:
    """Create a SIFT-like keypoint for testing."""

    return cv2.KeyPoint(
        x=x,
        y=y,
        size=size,
        angle=angle,
    )


def test_hough_vote_without_rotation() -> None:
    """Test center prediction when scale and rotation are unchanged."""

    template_keypoints = [
        create_keypoint(
            x=50.0,
            y=50.0,
            size=10.0,
            angle=0.0,
        )
    ]

    scene_keypoints = [
        create_keypoint(
            x=150.0,
            y=150.0,
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

    hough = GeneralizedHough()

    votes = hough.generate_votes(
        matches=matches,
        scene_keypoints=scene_keypoints,
        template_keypoints=template_keypoints,
        template_center=(100.0, 100.0),
    )

    assert len(votes) == 1

    vote = votes[0]

    assert vote.x == pytest.approx(200.0)
    assert vote.y == pytest.approx(200.0)
    assert vote.scale == pytest.approx(1.0)
    assert vote.angle == pytest.approx(0.0)


def test_hough_vote_with_scale() -> None:
    """Test center prediction when the object is scaled."""

    template_keypoints = [
        create_keypoint(
            x=50.0,
            y=50.0,
            size=10.0,
            angle=0.0,
        )
    ]

    scene_keypoints = [
        create_keypoint(
            x=200.0,
            y=200.0,
            size=20.0,
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

    hough = GeneralizedHough()

    votes = hough.generate_votes(
        matches=matches,
        scene_keypoints=scene_keypoints,
        template_keypoints=template_keypoints,
        template_center=(100.0, 100.0),
    )

    assert len(votes) == 1

    vote = votes[0]

    # Scale = 20 / 10 = 2.
    #
    # Template vector:
    # center - feature = (50, 50)
    #
    # Scaled vector:
    # (50, 50) * 2 = (100, 100)
    #
    # Predicted center:
    # (200, 200) + (100, 100) = (300, 300)

    assert vote.x == pytest.approx(300.0)
    assert vote.y == pytest.approx(300.0)
    assert vote.scale == pytest.approx(2.0)


def test_hough_vote_with_rotation() -> None:
    """Test center prediction when the object is rotated."""

    template_keypoints = [
        create_keypoint(
            x=50.0,
            y=100.0,
            size=10.0,
            angle=0.0,
        )
    ]

    scene_keypoints = [
        create_keypoint(
            x=200.0,
            y=200.0,
            size=10.0,
            angle=90.0,
        )
    ]

    matches = [
        FeatureMatch(
            scene_index=0,
            template_index=0,
            distance=10.0,
        )
    ]

    hough = GeneralizedHough()

    votes = hough.generate_votes(
        matches=matches,
        scene_keypoints=scene_keypoints,
        template_keypoints=template_keypoints,
        template_center=(100.0, 100.0),
    )

    assert len(votes) == 1

    vote = votes[0]

    # Template vector:
    #
    # center - feature = (50, 0)
    #
    # Rotate by 90 degrees:
    #
    # (50, 0) -> (0, 50)
    #
    # Predicted center:
    #
    # (200, 200) + (0, 50) = (200, 250)

    assert vote.x == pytest.approx(200.0)
    assert vote.y == pytest.approx(250.0)
    assert vote.scale == pytest.approx(1.0)
    assert vote.angle == pytest.approx(90.0)


def test_hough_clustering() -> None:
    """Test that votes in the same 4D bin are clustered together."""

    hough = GeneralizedHough(
        position_bin_size=20.0,
        scale_bin_size=0.1,
        angle_bin_size=15.0,
    )

    match_1 = FeatureMatch(
        scene_index=0,
        template_index=0,
        distance=10.0,
    )

    match_2 = FeatureMatch(
        scene_index=1,
        template_index=1,
        distance=12.0,
    )

    vote_1 = HoughVote(
        x=100.0,
        y=200.0,
        scale=1.0,
        angle=30.0,
        match=match_1,
    )

    vote_2 = HoughVote(
        x=105.0,
        y=205.0,
        scale=1.04,
        angle=34.0,
        match=match_2,
    )

    clusters = hough.cluster(
        [
            vote_1,
            vote_2,
        ]
    )

    assert len(clusters) == 1
    assert clusters[0].count == 2


def test_hough_different_bins_remain_separate() -> None:
    """Test that votes in different 4D bins remain separate."""

    hough = GeneralizedHough(
        position_bin_size=20.0,
        scale_bin_size=0.1,
        angle_bin_size=15.0,
    )

    match_1 = FeatureMatch(
        scene_index=0,
        template_index=0,
        distance=10.0,
    )

    match_2 = FeatureMatch(
        scene_index=1,
        template_index=1,
        distance=12.0,
    )

    vote_1 = HoughVote(
        x=100.0,
        y=100.0,
        scale=1.0,
        angle=30.0,
        match=match_1,
    )

    vote_2 = HoughVote(
        x=500.0,
        y=500.0,
        scale=2.0,
        angle=180.0,
        match=match_2,
    )

    clusters = hough.cluster(
        [
            vote_1,
            vote_2,
        ]
    )

    assert len(clusters) == 2

    assert clusters[0].count == 1
    assert clusters[1].count == 1


def test_hough_clusters_are_sorted_by_vote_count() -> None:
    """Test that strongest Hough clusters appear first."""

    hough = GeneralizedHough(
        position_bin_size=20.0,
        scale_bin_size=0.1,
        angle_bin_size=15.0,
    )

    match = FeatureMatch(
        scene_index=0,
        template_index=0,
        distance=10.0,
    )

    votes = [
        HoughVote(
            x=100.0,
            y=100.0,
            scale=1.0,
            angle=0.0,
            match=match,
        ),
        HoughVote(
            x=105.0,
            y=105.0,
            scale=1.0,
            angle=5.0,
            match=match,
        ),
        HoughVote(
            x=110.0,
            y=110.0,
            scale=1.0,
            angle=8.0,
            match=match,
        ),
        HoughVote(
            x=500.0,
            y=500.0,
            scale=2.0,
            angle=180.0,
            match=match,
        ),
    ]

    clusters = hough.cluster(votes)

    assert len(clusters) == 2
    assert clusters[0].count == 3
    assert clusters[1].count == 1


def test_hough_empty_votes() -> None:
    """Test that clustering an empty vote list returns no clusters."""

    hough = GeneralizedHough()

    clusters = hough.cluster([])

    assert clusters == []


def test_hough_bin_index() -> None:
    """Test conversion from continuous parameters to bin indices."""

    hough = GeneralizedHough(
        position_bin_size=20.0,
        scale_bin_size=0.1,
        angle_bin_size=15.0,
    )

    match = FeatureMatch(
        scene_index=0,
        template_index=0,
        distance=10.0,
    )

    vote = HoughVote(
        x=45.0,
        y=65.0,
        scale=1.25,
        angle=37.0,
        match=match,
    )

    bin_index = hough._get_bin_index(vote)

    assert bin_index == (
        2,   # floor(45 / 20)
        3,   # floor(65 / 20)
        12,  # floor(1.25 / 0.1)
        2,   # floor(37 / 15)
    )
