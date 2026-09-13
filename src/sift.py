from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias, cast

import cv2
import numpy as np
import numpy.typing as npt

DescriptorArray: TypeAlias = npt.NDArray[np.float32]


class _SIFT(Protocol):
    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: None,
    ) -> tuple[Sequence[cv2.KeyPoint], DescriptorArray | None]: ...


class SIFTExtractor:
    """Extract SIFT keypoints and descriptors from images."""

    def __init__(
        self,
        n_features: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ) -> None:
        sift_factory = cast(Callable[..., _SIFT], cv2.SIFT_create)  # type: ignore
        self.sift: _SIFT = sift_factory(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    def extract(
        self,
        image: np.ndarray | None,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Extract SIFT keypoints and descriptors from an image."""

        if image is None:
            raise ValueError("Input image cannot be None.")

        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        if descriptors is None:
            raise RuntimeError("No descriptors were computed.")

        if descriptors.shape[1] != 128:
            raise RuntimeError(
                f"Expected 128-dimensional SIFT descriptors, got {descriptors.shape[1]}."
            )

        return list(keypoints), descriptors
