# File: C:\Users\Arafat\Desktop\IMA Project\src\preprocessing.py

"""
In this file I want to put functions that prepare images before
feature extraction: resizing, grayscale conversion, denoising,
contrast enhancement, etc.
"""

import os
from typing import Tuple

import numpy as np
from skimage import io, color, transform, exposure, filters, img_as_float
import cv2

from . import config
from .utils import print_array_info


def load_image_as_float(image_path: str) -> np.ndarray:
    """
    In this function I want to read an image from disk and convert it
    to float in the range [0, 1]. I will use skimage for this.
    """
    temp_img = io.imread(image_path)
    temp_img = img_as_float(temp_img)  # this puts values in [0,1]
    return temp_img


def resize_image(temp_img: np.ndarray, output_size: Tuple[int, int] = None) -> np.ndarray:
    """
    In this function I want to resize the image to (height, width) that I use
    in the project. If no size is given, I use values from config.
    """
    if output_size is None:
        output_size = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    temp_resized = transform.resize(temp_img, output_size, anti_aliasing=True)
    return temp_resized


def to_grayscale(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to convert an RGB image to grayscale.
    If the image is already 2D, I just return it.
    """
    if temp_img.ndim == 2:
        return temp_img
    temp_gray = color.rgb2gray(temp_img)
    return temp_gray


def apply_gaussian_blur(temp_img: np.ndarray, sigma_value: float = 1.0) -> np.ndarray:
    """
    In this function I want to apply a Gaussian blur just to reduce noise a bit.
    """
    temp_blurred = filters.gaussian(temp_img, sigma=sigma_value, channel_axis=-1 if temp_img.ndim == 3 else None)
    return temp_blurred


def apply_median_filter(temp_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    In this function I want to apply a median filter, which is good for
    removing salt-and-pepper noise.
    """
    if temp_img.ndim == 3:
        # I convert to grayscale first for median (simpler)
        temp_gray = to_grayscale(temp_img)
    else:
        temp_gray = temp_img
    temp_filtered = filters.median(temp_gray, footprint=np.ones((kernel_size, kernel_size)))
    return temp_filtered


def apply_clahe_on_gray(temp_gray: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """
    In this function I want to improve local contrast using CLAHE
    (Contrast Limited Adaptive Histogram Equalization) on a grayscale image.
    """
    temp_clahe = exposure.equalize_adapthist(temp_gray, clip_limit=clip_limit)
    return temp_clahe


def apply_gamma_correction(temp_img: np.ndarray, gamma_value: float = 1.0) -> np.ndarray:
    """
    In this function I want to change brightness using gamma correction.
    gamma < 1 makes the image brighter, gamma > 1 makes it darker.
    """
    temp_corrected = exposure.adjust_gamma(temp_img, gamma=gamma_value)
    return temp_corrected


def rgb_to_hsv(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to convert RGB image to HSV color space.
    """
    if temp_img.ndim == 2:
        # if already gray, I stack to fake RGB then convert
        temp_img = np.stack([temp_img, temp_img, temp_img], axis=-1)
    temp_hsv = color.rgb2hsv(temp_img)
    return temp_hsv


def rgb_to_lab(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to convert RGB image to Lab color space,
    which sometimes is better for color differences.
    """
    if temp_img.ndim == 2:
        temp_img = np.stack([temp_img, temp_img, temp_img], axis=-1)
    temp_lab = color.rgb2lab(temp_img)
    return temp_lab


def apply_bilateral_filter(temp_img: np.ndarray, sigma_color: float = 0.05, sigma_space: float = 15):
    """
    In this function I want to apply bilateral filtering which smooths the image
    but also tries to keep edges. I will use OpenCV for this.
    """
    if temp_img.ndim == 2:
        temp_img_cv = (temp_img * 255).astype(np.uint8)
        temp_filtered = cv2.bilateralFilter(temp_img_cv, d=9, sigmaColor=sigma_color * 255, sigmaSpace=sigma_space)
        temp_filtered = temp_filtered.astype(np.float32) / 255.0
    else:
        temp_img_cv = (temp_img * 255).astype(np.uint8)
        temp_filtered = cv2.bilateralFilter(temp_img_cv, d=9, sigmaColor=sigma_color * 255, sigmaSpace=sigma_space)
        temp_filtered = temp_filtered.astype(np.float32) / 255.0
    return temp_filtered


def full_preprocess_pipeline(
    image_path: str,
    use_gaussian: bool = True,
    use_median: bool = False,
    use_clahe: bool = True,
    gamma_value: float = 1.0,
) -> np.ndarray:
    """
    In this function I want to put a simple full preprocessing pipeline together.
    This is just one possible pipeline (later I may define others in pipelines.py).
    Steps:
    1) load image
    2) resize
    3) maybe gaussian blur
    4) convert to gray
    5) maybe CLAHE
    6) maybe gamma correction
    """
    temp_img = load_image_as_float(image_path)
    temp_img = resize_image(temp_img)

    if use_gaussian:
        temp_img = apply_gaussian_blur(temp_img, sigma_value=1.0)

    temp_gray = to_grayscale(temp_img)

    if use_median:
        temp_gray = apply_median_filter(temp_gray, kernel_size=3)

    if use_clahe:
        temp_gray = apply_clahe_on_gray(temp_gray, clip_limit=0.01)

    if gamma_value != 1.0:
        temp_gray = apply_gamma_correction(temp_gray, gamma_value=gamma_value)

    # for debugging I might need to see stats of the final gray image
    # print_array_info(temp_gray, name="preprocessed_gray")

    return temp_gray


if __name__ == "__main__":
    # Small test to see if things work (I need at least one image in RAW_DATA_DIR)
    from . import config as cfg

    cfg.make_all_dirs()
    sample_folder = cfg.RAW_DATA_DIR
    temp_files = []
    for root, dirs, files in os.walk(sample_folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                temp_files.append(os.path.join(root, f))
        break  # just first folder level

    if len(temp_files) > 0:
        test_path = temp_files[0]
        print("Testing preprocessing on:", test_path)
        img = full_preprocess_pipeline(test_path)
        print_array_info(img, "preprocessed_gray")
    else:
        print("No images found in RAW_DATA_DIR to test preprocessing.")
