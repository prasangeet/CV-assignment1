from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path("data")

INPUT_IMAGES = [
    DATA_DIR / "1.jpg",
    DATA_DIR / "2.jpg",
    DATA_DIR / "3.jpg",
    DATA_DIR / "4.jpg",
]

OUTPUT_IMAGE = DATA_DIR / "scene.jpg"


def load_images(paths: list[Path]) -> list[np.ndarray]:
    """Load all input images."""

    images = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        images.append(image)

    return images


def resize_images(
    images: list[np.ndarray],
) -> list[np.ndarray]:
    """Resize all images to the dimensions of the smallest image."""

    target_width = min(image.shape[1] for image in images)

    target_height = min(image.shape[0] for image in images)

    return [
        cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]


def create_grid(
    images: list[np.ndarray],
) -> np.ndarray:
    """Create a 2x2 grid from four images."""

    if len(images) != 4:
        raise ValueError("Exactly four images are required.")

    top_row = cv2.hconcat([images[0], images[1]])

    bottom_row = cv2.hconcat([images[2], images[3]])

    return cv2.vconcat([top_row, bottom_row])


def main() -> None:
    """Create a 2x2 scene image from images 1 through 4."""

    images = load_images(INPUT_IMAGES)

    images = resize_images(images)

    scene = create_grid(images)

    success = cv2.imwrite(
        str(OUTPUT_IMAGE),
        scene,
    )

    if not success:
        raise OSError(f"Failed to save image: {OUTPUT_IMAGE}")

    print(f"Created: {OUTPUT_IMAGE}")
    print(f"Scene size: {scene.shape[1]} x {scene.shape[0]}")


if __name__ == "__main__":
    main()
