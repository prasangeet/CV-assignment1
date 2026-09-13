from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """A detected object instance with optional oriented corners."""

    box: tuple[float, float, float, float]
    score: float
    corners: tuple[tuple[float, float], ...] | None = None


class NonMaximumSuppression:
    """Suppress duplicate detections using IoU."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        self.iou_threshold = iou_threshold

    @staticmethod
    def iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        """Calculate Intersection over Union between two boxes."""

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0.0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0.0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = intersection_width * intersection_height

        area_a = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union_area = area_a + area_b - intersection_area

        if union_area == 0.0:
            return 0.0

        return intersection_area / union_area

    def suppress(
        self,
        detections: list[Detection],
    ) -> list[Detection]:
        """Remove overlapping duplicate detections."""

        if not detections:
            return []

        remaining = sorted(
            detections,
            key=lambda detection: detection.score,
            reverse=True,
        )

        selected: list[Detection] = []

        while remaining:
            best = remaining.pop(0)

            selected.append(best)

            remaining = [
                detection
                for detection in remaining
                if self.iou(
                    best.box,
                    detection.box,
                )
                < self.iou_threshold
            ]

        return selected
