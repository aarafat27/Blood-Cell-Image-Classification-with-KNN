# File: C:\Users\Arafat\Desktop\IMA Project\src\features.py

"""
In this file I want to convert images into feature vectors.
I will use the preprocessing functions and the filters to build
different feature banks: raw pixels, histograms, edges, texture,
frequency, morphology, etc.

The idea is each function here returns a 1D numpy array (feature vector)
for one image. Then I also create a function that builds a full
feature matrix for a list of images and a list of feature types.
"""

from typing import List, Dict, Callable

import numpy as np
from skimage import color
from skimage.exposure import histogram
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from . import config
from . import preprocessing
from . import filters as my_filters
from .utils import print_array_info


# BASIC FEATURES

def feature_raw_pixels(image_path: str) -> np.ndarray:
    """
    In this function I want to get raw pixel values as features.
    Steps: load image -> resize -> flatten (grayscale).
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    temp_gray = preprocessing.to_grayscale(temp_img)
    temp_flat = temp_gray.flatten()
    return temp_flat.astype(np.float32)


def feature_gray_hist(image_path: str, num_bins: int = 32) -> np.ndarray:
    """
    In this function I want to get a grayscale intensity histogram.
    This should roughly describe the brightness distribution.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    temp_gray = preprocessing.to_grayscale(temp_img)
    hist_values, hist_centers = histogram(temp_gray, nbins=num_bins)
    hist_values = hist_values.astype(float)
    # normalize to sum 1
    total = hist_values.sum()
    if total > 0:
        hist_values = hist_values / total
    return hist_values.astype(np.float32)


def feature_color_hist_rgb(image_path: str, num_bins: int = 16) -> np.ndarray:
    """
    In this function I want to compute separate histograms for R, G, B channels
    and then concatenate them.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    if temp_img.ndim == 2:
        temp_img = np.stack([temp_img, temp_img, temp_img], axis=-1)

    feature_list = []
    for temp_channel in range(3):
        channel_data = temp_img[:, :, temp_channel]
        hist_values, hist_centers = histogram(channel_data, nbins=num_bins)
        hist_values = hist_values.astype(float)
        total = hist_values.sum()
        if total > 0:
            hist_values = hist_values / total
        feature_list.append(hist_values)

    feature_vec = np.concatenate(feature_list).astype(np.float32)
    return feature_vec


def feature_color_hist_hsv(image_path: str, num_bins: int = 16) -> np.ndarray:
    """
    In this function I want to compute histograms in HSV color space,
    which sometimes is better for color description.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    temp_hsv = preprocessing.rgb_to_hsv(temp_img)

    feature_list = []
    for temp_channel in range(3):
        channel_data = temp_hsv[:, :, temp_channel]
        hist_values, hist_centers = histogram(channel_data, nbins=num_bins)
        hist_values = hist_values.astype(float)
        total = hist_values.sum()
        if total > 0:
            hist_values = hist_values / total
        feature_list.append(hist_values)

    feature_vec = np.concatenate(feature_list).astype(np.float32)
    return feature_vec


# EDGE / TEXTURE FEATURES

def feature_sobel_hist(image_path: str, num_bins: int = 32) -> np.ndarray:
    """
    In this function I want to compute a histogram of Sobel edge magnitudes.
    This gives me a sense of how strong edges are in the image.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    sobel_img = my_filters.sobel_magnitude(temp_img)
    hist_values, hist_centers = histogram(sobel_img, nbins=num_bins)
    hist_values = hist_values.astype(float)
    total = hist_values.sum()
    if total > 0:
        hist_values = hist_values / total
    return hist_values.astype(np.float32)


def feature_lbp_hist(image_path: str, radius_value: int = 2, num_points: int = None) -> np.ndarray:
    """
    In this function I want to compute Local Binary Pattern (LBP) histogram.
    This is a classic texture descriptor.
    """
    if num_points is None:
        num_points = 8 * radius_value

    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    temp_gray = preprocessing.to_grayscale(temp_img)

    lbp_img = local_binary_pattern(temp_gray, P=num_points, R=radius_value, method="uniform")
    lbp_values = lbp_img.ravel()
    n_bins = int(lbp_values.max() + 2)
    hist_values, _ = np.histogram(lbp_values, bins=n_bins, range=(0, n_bins), density=True)
    return hist_values.astype(np.float32)


def feature_glcm_texture(image_path: str) -> np.ndarray:
    """
    In this function I want to compute some Haralick-like features
    using gray-level co-occurrence matrix (GLCM).
    I will use a quantized grayscale image to 16 levels.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    temp_gray = preprocessing.to_grayscale(temp_img)

    # quantize to 16 levels
    temp_gray_q = (temp_gray * 15).astype(np.uint8)

    distances = [1, 2]  # 1 and 2 pixel distances
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(temp_gray_q, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feature_list = []
    for prop in props:
        temp_prop = graycoprops(glcm, prop)
        feature_list.append(temp_prop.flatten())

    feature_vec = np.concatenate(feature_list).astype(np.float32)
    return feature_vec



# FREQUENCY FEATURES

def feature_fft_low_patch(image_path: str, patch_size: int = 15) -> np.ndarray:
    """
    In this function I want to get the low-frequency Fourier patch as a feature.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    patch = my_filters.fft_low_freq_patch(temp_img, patch_size=patch_size)
    feature_vec = patch.flatten().astype(np.float32)
    return feature_vec


def feature_fft_radial_energy(image_path: str, num_bins: int = 8) -> np.ndarray:
    """
    In this function I want to get the radial energy distribution from FFT.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    radial = my_filters.fft_radial_energy(temp_img, num_bins=num_bins)
    return radial.astype(np.float32)


def feature_dct_low_block(image_path: str, block_size: int = 8) -> np.ndarray:
    """
    In this function I want to get the low-frequency DCT block.
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    block = my_filters.dct_low_freq_block(temp_img, block_size=block_size)
    feature_vec = block.flatten().astype(np.float32)
    return feature_vec


# MORPHOLOGY / SHAPE FEATURES


def feature_shape_morphology(image_path: str) -> np.ndarray:
    """
    In this function I want to get simple shape features using morphology
    (area, perimeter, eccentricity etc.).
    """
    temp_img = preprocessing.load_image_as_float(image_path)
    temp_img = preprocessing.resize_image(temp_img)
    shape_vec = my_filters.shape_features_from_binary(temp_img)
    return shape_vec.astype(np.float32)



# FEATURE MATRIX BUILDER

# here I prepare a dictionary to map feature names (strings) to functions
FEATURE_FUNCTIONS: Dict[str, Callable[[str], np.ndarray]] = {
    "raw_pixels": feature_raw_pixels,
    "gray_hist": feature_gray_hist,
    "rgb_hist": feature_color_hist_rgb,
    "hsv_hist": feature_color_hist_hsv,
    "sobel_hist": feature_sobel_hist,
    "lbp_hist": feature_lbp_hist,
    "glcm_texture": feature_glcm_texture,
    "fft_low_patch": feature_fft_low_patch,
    "fft_radial": feature_fft_radial_energy,
    "dct_low_block": feature_dct_low_block,
    "shape_morphology": feature_shape_morphology,
}


def build_feature_matrix(image_path_list: List[str], feature_name_list: List[str]) -> np.ndarray:
    """
    In this function I want to build a big feature matrix from a list of images.
    For each image, I will compute all the requested features (in feature_name_list),
    then concatenate them into one long vector.
    """
    print("[features] Building feature matrix...")
    print("[features] Number of images:", len(image_path_list))
    print("[features] Feature types:", feature_name_list)

    # First I compute the length of the concatenated feature vector
    temp_first_vec_list = []
    first_image_path = image_path_list[0]
    for temp_name in feature_name_list:
        temp_func = FEATURE_FUNCTIONS[temp_name]
        temp_vec = temp_func(first_image_path)
        temp_first_vec_list.append(temp_vec)
    first_concat = np.concatenate(temp_first_vec_list)
    feature_dim = first_concat.shape[0]

    print(f"[features] Feature dimension per image: {feature_dim}")

    num_images = len(image_path_list)
    feature_matrix = np.zeros((num_images, feature_dim), dtype=np.float32)

    # Now I fill the matrix
    for idx, img_path in enumerate(image_path_list):
        temp_vec_list = []
        for temp_name in feature_name_list:
            temp_func = FEATURE_FUNCTIONS[temp_name]
            temp_vec = temp_func(img_path)
            temp_vec_list.append(temp_vec)
        temp_concat = np.concatenate(temp_vec_list)
        feature_matrix[idx, :] = temp_concat

        if (idx + 1) % 20 == 0 or (idx + 1) == num_images:
            print(f"[features] Processed {idx + 1}/{num_images} images")

    print_array_info(feature_matrix, "feature_matrix")
    return feature_matrix
