from pathlib import Path

import cv2
import numpy as np

from .matcher import FeatureMatch
from .nms import Detection


class Visualizer:
    """Create visualizations for the object detection pipeline."""

    @staticmethod
    def draw_matches(
        template_image: np.ndarray,
        scene_image: np.ndarray,
        template_keypoints: list[cv2.KeyPoint],
        scene_keypoints: list[cv2.KeyPoint],
        matches: list[FeatureMatch],
    ) -> np.ndarray:
        """Draw all feature correspondences between template and scene."""

        cv2_matches = [
            cv2.DMatch(
                _queryIdx=match.template_index,
                _trainIdx=match.scene_index,
                _distance=match.distance,
            )
            for match in matches
        ]

        return cv2.drawMatches(
            template_image,
            template_keypoints,
            scene_image,
            scene_keypoints,
            cv2_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    @staticmethod
    def draw_inlier_matches(
        template_image: np.ndarray,
        scene_image: np.ndarray,
        template_keypoints: list[cv2.KeyPoint],
        scene_keypoints: list[cv2.KeyPoint],
        matches: list[FeatureMatch],
    ) -> np.ndarray:
        """Draw only geometrically verified feature correspondences."""

        cv2_matches = [
            cv2.DMatch(
                _queryIdx=match.template_index,
                _trainIdx=match.scene_index,
                _distance=match.distance,
            )
            for match in matches
        ]

        return cv2.drawMatches(
            template_image,
            template_keypoints,
            scene_image,
            scene_keypoints,
            cv2_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw oriented detection outlines when projected corners are available."""

        output = image.copy()

        for index, detection in enumerate(detections, start=1):
            if detection.corners is not None:
                polygon = np.rint(
                    np.array(detection.corners, dtype=np.float64)
                ).astype(np.int32).reshape((-1, 1, 2))

                cv2.polylines(
                    output,
                    [polygon],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                label_x, label_y = min(
                    detection.corners,
                    key=lambda corner: corner[1],
                )
            else:
                x1, y1, x2, y2 = detection.box

                top_left = (
                    int(round(x1)),
                    int(round(y1)),
                )

                bottom_right = (
                    int(round(x2)),
                    int(round(y2)),
                )

                cv2.rectangle(
                    output,
                    top_left,
                    bottom_right,
                    (0, 255, 0),
                    2,
                )

                label_x, label_y = top_left

            label = f"Object {index} ({detection.score:.2f})"

            label_origin = (
                int(round(label_x)),
                max(20, int(round(label_y)) - 8),
            )

            cv2.putText(
                output,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output

    @staticmethod
    def save(
        image: np.ndarray,
        path: str | Path,
    ) -> None:
        """Save a visualization to disk."""

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            raise OSError(f"Failed to save image to {output_path}.")
