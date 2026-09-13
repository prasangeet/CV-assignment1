from dataclasses import dataclass

import cv2
import numpy as np

from src.matcher import FeatureMatch


@dataclass(frozen=True)
class HoughVote:
    """A single vote in the 4D Hough parameter space."""

    x: float
    y: float
    scale: float
    angle: float
    match: FeatureMatch


@dataclass
class HoughCluster:
    """A group of votes occupying the same 4D Hough bin."""

    bin_index: tuple[int, int, int, int]
    votes: list[HoughVote]

    @property
    def count(self) -> int:
        """Return the number of votes in the cluster."""

        return len(self.votes)


class GeneralizedHough:
    """4D Generalized Hough Transform for object-instance localization."""

    def __init__(
        self,
        position_bin_size: float = 20.0,
        scale_bin_size: float = 0.1,
        angle_bin_size: float = 15.0,
    ) -> None:
        if position_bin_size <= 0:
            raise ValueError(
                "position_bin_size must be positive."
            )

        if scale_bin_size <= 0:
            raise ValueError(
                "scale_bin_size must be positive."
            )

        if angle_bin_size <= 0:
            raise ValueError(
                "angle_bin_size must be positive."
            )

        if angle_bin_size > 360.0:
            raise ValueError(
                "angle_bin_size cannot exceed 360 degrees."
            )

        self.position_bin_size = position_bin_size
        self.scale_bin_size = scale_bin_size
        self.angle_bin_size = angle_bin_size

    def generate_votes(
        self,
        matches: list[FeatureMatch],
        scene_keypoints: list[cv2.KeyPoint],
        template_keypoints: list[cv2.KeyPoint],
        template_center: tuple[float, float],
    ) -> list[HoughVote]:
        """Generate 4D Hough votes from feature correspondences."""

        votes: list[HoughVote] = []

        center_x, center_y = template_center

        for match in matches:
            scene_point = scene_keypoints[
                match.scene_index
            ]

            template_point = template_keypoints[
                match.template_index
            ]

            if template_point.size <= 0:
                continue

            if scene_point.size <= 0:
                continue

            scale = (
                scene_point.size
                / template_point.size
            )

            angle = (
                scene_point.angle
                - template_point.angle
            ) % 360.0

            template_dx = (
                center_x
                - template_point.pt[0]
            )

            template_dy = (
                center_y
                - template_point.pt[1]
            )

            theta = np.deg2rad(angle)

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            rotated_dx = (
                cos_theta * template_dx
                - sin_theta * template_dy
            )

            rotated_dy = (
                sin_theta * template_dx
                + cos_theta * template_dy
            )

            predicted_x = (
                scene_point.pt[0]
                + scale * rotated_dx
            )

            predicted_y = (
                scene_point.pt[1]
                + scale * rotated_dy
            )

            votes.append(
                HoughVote(
                    x=float(predicted_x),
                    y=float(predicted_y),
                    scale=float(scale),
                    angle=float(angle),
                    match=match,
                )
            )

        return votes

    def cluster(
        self,
        votes: list[HoughVote],
    ) -> list[HoughCluster]:
        """
        Accumulate votes into the 4D Hough parameter space.

        Each vote is assigned to exactly one bin. Bins with many
        votes represent probable object instances.
        """

        if not votes:
            return []

        accumulator: dict[
            tuple[int, int, int, int],
            list[HoughVote],
        ] = {}

        for vote in votes:
            bin_index = self._get_bin_index(vote)

            if bin_index not in accumulator:
                accumulator[bin_index] = []

            accumulator[bin_index].append(vote)

        clusters = [
            HoughCluster(
                bin_index=bin_index,
                votes=cluster_votes,
            )
            for bin_index, cluster_votes
            in accumulator.items()
        ]

        clusters.sort(
            key=lambda cluster: cluster.count,
            reverse=True,
        )

        return clusters

    def _get_bin_index(
        self,
        vote: HoughVote,
    ) -> tuple[int, int, int, int]:
        """Convert a continuous vote into a discrete 4D bin."""

        x_bin = int(
            np.floor(
                vote.x / self.position_bin_size
            )
        )

        y_bin = int(
            np.floor(
                vote.y / self.position_bin_size
            )
        )

        scale_bin = int(
            np.floor(
                vote.scale / self.scale_bin_size
            )
        )

        angle_bin = int(
            np.floor(
                vote.angle / self.angle_bin_size
            )
        )

        return (
            x_bin,
            y_bin,
            scale_bin,
            angle_bin,
        )
