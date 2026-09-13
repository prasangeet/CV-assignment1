from pathlib import Path

import cv2
import numpy as np

from src.affine import AffineTransform
from src.detection import DetectionProjector
from src.hough import GeneralizedHough, HoughCluster
from src.matcher import DescriptorMatcher, FeatureMatch
from src.nms import Detection, NonMaximumSuppression
from src.ransac import AffineRANSAC, RANSACResult
from src.sift import SIFTExtractor
from src.visualization import Visualizer


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

TEMPLATE_PATH = DATA_DIR / "template.jpg"
SCENE_PATH = DATA_DIR / "scene.jpg"

NAIVE_MATCHES_PATH = OUTPUT_DIR / "naive_matches.jpg"
VERIFIED_MATCHES_PATH = OUTPUT_DIR / "verified_matches.jpg"
FINAL_DETECTIONS_PATH = OUTPUT_DIR / "final_detections.jpg"


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

MAX_IMAGE_SIZE = 2560
MAX_SIFT_FEATURES = 6000

RATIO_THRESHOLD = 0.75


# ---------------------------------------------------------
# Hough Transform
# ---------------------------------------------------------

POSITION_BIN_SIZE = 20.0
SCALE_BIN_SIZE = 0.1
ANGLE_BIN_SIZE = 15.0

MIN_HOUGH_VOTES = 3


# ---------------------------------------------------------
# RANSAC
# ---------------------------------------------------------

RANSAC_ITERATIONS = 500
RANSAC_THRESHOLD = 5.0
MIN_RANSAC_INLIERS = 3


# ---------------------------------------------------------
# NMS
# ---------------------------------------------------------

NMS_IOU_THRESHOLD = 0.5


class ObjectDetector:
    """Detect multiple instances of a template in a scene."""

    def __init__(self) -> None:
        self.sift = SIFTExtractor(
            n_features=MAX_SIFT_FEATURES,
        )

        self.matcher = DescriptorMatcher(
            ratio_threshold=RATIO_THRESHOLD,
        )

        self.hough = GeneralizedHough(
            position_bin_size=POSITION_BIN_SIZE,
            scale_bin_size=SCALE_BIN_SIZE,
            angle_bin_size=ANGLE_BIN_SIZE,
        )

        self.ransac = AffineRANSAC(
            iterations=RANSAC_ITERATIONS,
            reprojection_threshold=RANSAC_THRESHOLD,
            min_inliers=MIN_RANSAC_INLIERS,
        )

        self.projector = DetectionProjector()

        self.nms = NonMaximumSuppression(
            iou_threshold=NMS_IOU_THRESHOLD,
        )

        self.visualizer = Visualizer()

    def detect(
        self,
        template_image: np.ndarray,
        scene_image: np.ndarray,
    ) -> tuple[
        list[Detection],
        list[cv2.KeyPoint],
        list[cv2.KeyPoint],
        list[FeatureMatch],
        list[FeatureMatch],
    ]:
        """Run feature extraction, matching, and detection."""

        # -------------------------------------------------
        # SIFT feature extraction
        # -------------------------------------------------

        template_keypoints, template_descriptors = (
            self.sift.extract(template_image)
        )

        scene_keypoints, scene_descriptors = (
            self.sift.extract(scene_image)
        )

        print(
            f"Template keypoints: "
            f"{len(template_keypoints)}"
        )

        print(
            f"Scene keypoints: "
            f"{len(scene_keypoints)}"
        )

        # -------------------------------------------------
        # Descriptor matching
        # -------------------------------------------------

        matches = self.matcher.match(
            scene_descriptors=scene_descriptors,
            template_descriptors=template_descriptors,
        )

        print(
            f"Valid matches: {len(matches)}"
        )

        # -------------------------------------------------
        # Instance detection
        # -------------------------------------------------

        detections, verified_matches = self._find_instances(
            template_keypoints=template_keypoints,
            scene_keypoints=scene_keypoints,
            matches=matches,
            template_width=template_image.shape[1],
            template_height=template_image.shape[0],
        )

        return (
            detections,
            template_keypoints,
            scene_keypoints,
            matches,
            verified_matches,
        )

    def _find_instances(
        self,
        template_keypoints: list[cv2.KeyPoint],
        scene_keypoints: list[cv2.KeyPoint],
        matches: list[FeatureMatch],
        template_width: int,
        template_height: int,
    ) -> tuple[list[Detection], list[FeatureMatch]]:
        """Find all object instances using iterative extraction."""

        template_center = (
            template_width / 2.0,
            template_height / 2.0,
        )

        active_matches = matches.copy()

        detections: list[Detection] = []

        verified_matches: list[FeatureMatch] = []

        iteration = 1

        # -------------------------------------------------
        # Greedy extraction loop
        # -------------------------------------------------

        while len(active_matches) >= MIN_HOUGH_VOTES:

            print(
                f"\nDetection iteration {iteration}"
            )

            # ---------------------------------------------
            # Generate Hough votes
            # ---------------------------------------------

            votes = self.hough.generate_votes(
                matches=active_matches,
                scene_keypoints=scene_keypoints,
                template_keypoints=template_keypoints,
                template_center=template_center,
            )

            if len(votes) < MIN_HOUGH_VOTES:
                break

            # ---------------------------------------------
            # Cluster votes in 4D parameter space
            # ---------------------------------------------

            clusters = self.hough.cluster(votes)

            # ---------------------------------------------
            # Find a geometrically valid cluster
            # ---------------------------------------------

            candidate = self._find_valid_cluster(
                clusters=clusters,
                template_keypoints=template_keypoints,
                scene_keypoints=scene_keypoints,
            )

            if candidate is None:
                print(
                    "No valid object cluster found."
                )
                break

            cluster, ransac_result = candidate

            # ---------------------------------------------
            # Collect RANSAC inliers
            # ---------------------------------------------

            inlier_matches = self._get_inlier_matches(
                cluster=cluster,
                ransac_result=ransac_result,
            )

            verified_matches.extend(inlier_matches)

            # ---------------------------------------------
            # Project template through affine transform
            # ---------------------------------------------

            projected = self.projector.project(
                transform=ransac_result.transform,
                template_width=template_width,
                template_height=template_height,
            )

            detection = self.projector.to_detection(
                projected=projected,
                score=float(
                    ransac_result.inlier_count
                ),
            )

            detections.append(detection)

            print(
                f"Detected instance "
                f"{len(detections)} "
                f"with "
                f"{ransac_result.inlier_count} "
                f"inliers."
            )

            # ---------------------------------------------
            # Remove this object's inliers
            # ---------------------------------------------

            active_matches = self._remove_inliers(
                active_matches=active_matches,
                inlier_matches=set(inlier_matches),
            )

            iteration += 1

        return detections, verified_matches

    def _find_valid_cluster(
        self,
        clusters: list[HoughCluster],
        template_keypoints: list[cv2.KeyPoint],
        scene_keypoints: list[cv2.KeyPoint],
    ) -> tuple[HoughCluster, RANSACResult] | None:
        """Find the strongest cluster with a valid affine model."""

        for cluster in clusters:

            if cluster.count < MIN_HOUGH_VOTES:
                break

            # ---------------------------------------------
            # Build corresponding point arrays
            # ---------------------------------------------

            template_points = np.array(
                [
                    template_keypoints[
                        vote.match.template_index
                    ].pt
                    for vote in cluster.votes
                ],
                dtype=np.float64,
            )

            scene_points = np.array(
                [
                    scene_keypoints[
                        vote.match.scene_index
                    ].pt
                    for vote in cluster.votes
                ],
                dtype=np.float64,
            )

            # ---------------------------------------------
            # Estimate affine transform using RANSAC
            # ---------------------------------------------

            try:
                result = self.ransac.estimate(
                    template_points=template_points,
                    scene_points=scene_points,
                )
            except (RuntimeError, ValueError):
                continue

            # ---------------------------------------------
            # Reject geometrically invalid transforms
            # ---------------------------------------------

            if not self._is_valid_transform(
                result.transform
            ):
                continue

            return cluster, result

        return None

    @staticmethod
    def _get_inlier_matches(
        cluster: HoughCluster,
        ransac_result: RANSACResult,
    ) -> list[FeatureMatch]:
        """Return matches identified as RANSAC inliers."""

        return [
            vote.match
            for index, vote in enumerate(cluster.votes)
            if ransac_result.inlier_mask[index]
        ]

    @staticmethod
    def _remove_inliers(
        active_matches: list[FeatureMatch],
        inlier_matches: set[FeatureMatch],
    ) -> list[FeatureMatch]:
        """Remove matches belonging to a detected instance."""

        return [
            match
            for match in active_matches
            if match not in inlier_matches
        ]

    @staticmethod
    def _is_valid_transform(
        transform: AffineTransform,
    ) -> bool:
        """Reject degenerate or excessively sheared transforms."""

        linear_part = np.array(
            [
                [transform.a, transform.b],
                [transform.c, transform.d],
            ],
            dtype=np.float64,
        )

        # ---------------------------------------------
        # Check orientation
        # ---------------------------------------------

        determinant = float(
            np.linalg.det(linear_part)
        )

        if determinant <= 0.0:
            return False

        # ---------------------------------------------
        # Check singular values
        # ---------------------------------------------

        singular_values = np.linalg.svd(
            linear_part,
            compute_uv=False,
        )

        if np.any(singular_values <= 1e-6):
            return False

        # ---------------------------------------------
        # Estimate overall scale
        # ---------------------------------------------

        scale = float(
            np.sqrt(
                singular_values[0]
                * singular_values[1]
            )
        )

        # ---------------------------------------------
        # Check condition number
        # ---------------------------------------------

        condition_number = float(
            singular_values[0]
            / singular_values[1]
        )

        if not 0.25 <= scale <= 4.0:
            return False

        if condition_number > 3.0:
            return False

        return True


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk."""

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    return image


def resize_image(
    image: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, float]:
    """
    Resize an image while preserving its aspect ratio.

    Returns the resized image and the scale factor used.
    """

    height, width = image.shape[:2]

    largest_dimension = max(
        width,
        height,
    )

    if largest_dimension <= max_size:
        return image.copy(), 1.0

    scale = max_size / largest_dimension

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def scale_detection(
    detection: Detection,
    scale: float,
) -> Detection:
    """Convert a detection from resized to original coordinates."""

    if scale <= 0.0:
        raise ValueError(
            "scale must be positive."
        )

    x1, y1, x2, y2 = detection.box

    inverse_scale = 1.0 / scale

    original_box = (
        x1 * inverse_scale,
        y1 * inverse_scale,
        x2 * inverse_scale,
        y2 * inverse_scale,
    )

    original_corners = (
        tuple(
            (x * inverse_scale, y * inverse_scale)
            for x, y in detection.corners
        )
        if detection.corners is not None
        else None
    )

    return Detection(
        box=original_box,
        score=detection.score,
        corners=original_corners,
    )


def main() -> None:
    """Run the complete object detection pipeline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load original images
    # -----------------------------------------------------

    template_original = load_image(
        TEMPLATE_PATH
    )

    scene_original = load_image(
        SCENE_PATH
    )

    print(
        "Original template size: "
        f"{template_original.shape[1]} x "
        f"{template_original.shape[0]}"
    )

    print(
        "Original scene size: "
        f"{scene_original.shape[1]} x "
        f"{scene_original.shape[0]}"
    )

    # -----------------------------------------------------
    # Resize images for memory-safe processing
    # -----------------------------------------------------

    template_color, _ = resize_image(
        template_original,
        MAX_IMAGE_SIZE,
    )

    scene_color, scene_scale = resize_image(
        scene_original,
        MAX_IMAGE_SIZE,
    )

    print(
        "Processing template size: "
        f"{template_color.shape[1]} x "
        f"{template_color.shape[0]}"
    )

    print(
        "Processing scene size: "
        f"{scene_color.shape[1]} x "
        f"{scene_color.shape[0]}"
    )

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

    template_gray = cv2.cvtColor(
        template_color,
        cv2.COLOR_BGR2GRAY,
    )

    scene_gray = cv2.cvtColor(
        scene_color,
        cv2.COLOR_BGR2GRAY,
    )

    # -----------------------------------------------------
    # Run detector
    # -----------------------------------------------------

    detector = ObjectDetector()

    (
        detections,
        template_keypoints,
        scene_keypoints,
        matches,
        verified_matches,
    ) = detector.detect(
        template_image=template_gray,
        scene_image=scene_gray,
    )

    print(
        f"\nDetections before NMS: "
        f"{len(detections)}"
    )

    print(
        f"Verified matches: "
        f"{len(verified_matches)}"
    )

    # -----------------------------------------------------
    # NMS
    # -----------------------------------------------------

    final_detections = detector.nms.suppress(
        detections
    )

    print(
        f"Detections after NMS: "
        f"{len(final_detections)}"
    )

    # -----------------------------------------------------
    # Convert detections back to original coordinates
    # -----------------------------------------------------

    original_detections = [
        scale_detection(
            detection=detection,
            scale=scene_scale,
        )
        for detection in final_detections
    ]

    # -----------------------------------------------------
    # Naive matching visualization
    # -----------------------------------------------------

    naive_visualization = (
        detector.visualizer.draw_matches(
            template_image=template_color,
            scene_image=scene_color,
            template_keypoints=template_keypoints,
            scene_keypoints=scene_keypoints,
            matches=matches,
        )
    )

    detector.visualizer.save(
        image=naive_visualization,
        path=NAIVE_MATCHES_PATH,
    )

    # -----------------------------------------------------
    # RANSAC verified matching visualization
    # -----------------------------------------------------

    verified_visualization = (
        detector.visualizer.draw_inlier_matches(
            template_image=template_color,
            scene_image=scene_color,
            template_keypoints=template_keypoints,
            scene_keypoints=scene_keypoints,
            matches=verified_matches,
        )
    )

    detector.visualizer.save(
        image=verified_visualization,
        path=VERIFIED_MATCHES_PATH,
    )

    # -----------------------------------------------------
    # Final detection visualization
    # -----------------------------------------------------

    final_visualization = (
        detector.visualizer.draw_detections(
            image=scene_original,
            detections=original_detections,
        )
    )

    detector.visualizer.save(
        image=final_visualization,
        path=FINAL_DETECTIONS_PATH,
    )

    # -----------------------------------------------------
    # Print final detections
    # -----------------------------------------------------

    for index, detection in enumerate(
        original_detections,
        start=1,
    ):
        print(
            f"Object {index}: "
            f"box={detection.box}, "
            f"score={detection.score:.2f}"
        )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------

    print(
        f"\nSaved naive matching visualization to:"
        f"\n  {NAIVE_MATCHES_PATH}"
    )

    print(
        f"\nSaved verified matching visualization to:"
        f"\n  {VERIFIED_MATCHES_PATH}"
    )

    print(
        f"\nSaved final detections to:"
        f"\n  {FINAL_DETECTIONS_PATH}"
    )


if __name__ == "__main__":
    main()
