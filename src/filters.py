# File: C:\Users\Arafat\Desktop\IMA Project\src\filters.py

"""
In this file I want to put all the lower-level image processing filters.
So here I will do things like Sobel, Prewitt, Scharr, Laplacian,
Canny edges, Fourier transform, DCT, and some morphology operations.

The idea is that these functions work on numpy image arrays.
Later, the features.py file will use these to build feature vectors.
"""

from typing import Tuple, List

import numpy as np
from skimage import filters, feature, morphology, measure, color
from scipy.fft import fft2, fftshift, dctn



# EDGE AND GRADIENT FILTERS

def sobel_magnitude(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to compute the Sobel edge magnitude of an image.
    If the image is RGB, I convert it to grayscale first.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img
    temp_edges = filters.sobel(temp_gray)
    return temp_edges


def prewitt_magnitude(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to compute the Prewitt edge magnitude of an image.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img
    temp_edges = filters.prewitt(temp_gray)
    return temp_edges


def scharr_magnitude(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to compute the Scharr edge magnitude of an image.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img
    temp_edges = filters.scharr(temp_gray)
    return temp_edges


def laplacian_filter(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to apply a Laplacian filter to highlight areas
    of rapid intensity change.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img
    temp_edges = filters.laplace(temp_gray)
    return temp_edges


def log_edges(temp_img: np.ndarray, sigma_value: float = 1.0) -> np.ndarray:
    """
    In this function I want to apply Laplacian of Gaussian (LoG) filter,
    which is basically smoothing + second derivative.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img
    temp_edges = filters.laplace(filters.gaussian(temp_gray, sigma=sigma_value))
    return temp_edges


def canny_edges(temp_img: np.ndarray, sigma_value: float = 1.0) -> np.ndarray:
    """
    In this function I want to detect edges using Canny detector.
    This returns a binary edge map.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img
    temp_edges = feature.canny(temp_gray, sigma=sigma_value)
    return temp_edges.astype(float)



# FOURIER TRANSFORM STUFF


def fft_magnitude_spectrum(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to compute the 2D Fourier magnitude spectrum.
    Steps:
    1) convert to gray
    2) compute fft2
    3) shift with fftshift
    4) take magnitude and log scale
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img

    temp_fft = fft2(temp_gray)
    temp_fft_shifted = fftshift(temp_fft)
    temp_magnitude = np.abs(temp_fft_shifted)
    temp_log_mag = np.log1p(temp_magnitude)
    return temp_log_mag


def fft_low_freq_patch(temp_img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    In this function I want to take a small square patch around the
    center of the Fourier magnitude spectrum. This should represent
    low-frequency information.
    """
    mag = fft_magnitude_spectrum(temp_img)
    h, w = mag.shape
    center_row = h // 2
    center_col = w // 2
    half = patch_size // 2
    patch = mag[center_row - half:center_row + half, center_col - half:center_col + half]
    return patch


def fft_radial_energy(temp_img: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """
    In this function I want to compute a simple radial energy descriptor.
    I measure how much energy (magnitude) is in different frequency radii.
    """
    mag = fft_magnitude_spectrum(temp_img)
    h, w = mag.shape
    center_row = h // 2
    center_col = w // 2

    # create radius map
    y_indices, x_indices = np.indices(mag.shape)
    dist = np.sqrt((y_indices - center_row) ** 2 + (x_indices - center_col) ** 2)
    max_dist = dist.max()
    bin_edges = np.linspace(0, max_dist, num_bins + 1)

    radial_energy = np.zeros(num_bins, dtype=float)

    for i in range(num_bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if np.any(mask):
            radial_energy[i] = mag[mask].sum()
        else:
            radial_energy[i] = 0.0

    # normalize to sum 1 (avoid division by zero)
    total = radial_energy.sum()
    if total > 0:
        radial_energy = radial_energy / total

    return radial_energy


def dct_low_freq_block(temp_img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    In this function I want to compute a 2D DCT and keep only
    the low-frequency block (top-left corner).
    I use scipy's dctn for that.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img

    temp_dct = dctn(temp_gray, norm="ortho")
    block = temp_dct[:block_size, :block_size]
    return block



# MORPHOLOGY + SHAPE

def basic_morphology_ops(temp_img: np.ndarray, disk_size: int = 3) -> dict:
    """
    In this function I want to apply some morphology operators on a binary image.
    Steps:
    - if image is not binary, I will threshold it using Otsu
    - then I will do opening, closing, and morphological gradient
    and return all of them.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img

    # threshold to get binary
    thresh = filters.threshold_otsu(temp_gray)
    bin_img = temp_gray > thresh

    selem = morphology.disk(disk_size)

    opened = morphology.opening(bin_img, selem)
    closed = morphology.closing(bin_img, selem)
    dilated = morphology.dilation(bin_img, selem)
    eroded = morphology.erosion(bin_img, selem)
    morph_gradient = dilated.astype(float) - eroded.astype(float)

    result_dict = {
        "binary": bin_img.astype(float),
        "opened": opened.astype(float),
        "closed": closed.astype(float),
        "morph_gradient": morph_gradient,
    }
    return result_dict


def shape_features_from_binary(temp_img: np.ndarray) -> np.ndarray:
    """
    In this function I want to extract some simple shape features
    from a binary image. I will look at the biggest connected component
    and compute area, perimeter, eccentricity, etc.
    """
    if temp_img.ndim == 3:
        temp_gray = color.rgb2gray(temp_img)
    else:
        temp_gray = temp_img

    thresh = filters.threshold_otsu(temp_gray)
    bin_img = temp_gray > thresh

    labeled = measure.label(bin_img)
    regions = measure.regionprops(labeled)

    if len(regions) == 0:
        # no region found, I just return zeros
        return np.zeros(6, dtype=float)

    # I take the largest region
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    main_region = regions_sorted[0]

    area = main_region.area
    perimeter = main_region.perimeter
    eccentricity = main_region.eccentricity
    solidity = main_region.solidity
    extent = main_region.extent
    # simple roundness measure
    if perimeter > 0:
        roundness = 4 * np.pi * area / (perimeter ** 2)
    else:
        roundness = 0.0

    feature_vec = np.array([area, perimeter, eccentricity, solidity, extent, roundness], dtype=float)
    return feature_vec
