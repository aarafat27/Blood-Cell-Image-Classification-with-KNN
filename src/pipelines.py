# File: C:\Users\Arafat\Desktop\IMA Project\src\pipelines.py

"""
In this file I want to define different feature pipelines.
Each pipeline will have a name and a list of feature types to use.

I will not hard-code preprocessing here, because most preprocessing
is already inside the feature extraction functions in features.py.
But I will clearly define which feature combinations belong to which pipeline.
"""

from typing import Dict, List


# Here I define a dictionary where each key is the pipeline name
# and the value is another dict with the feature list.
PIPELINES: Dict[str, Dict[str, List[str]]] = {
    # 1) baseline: just raw pixels
    "RawPixels": {
        "description": "Simple baseline using raw grayscale pixel values.",
        "feature_names": ["raw_pixels"],
    },

    # 2) gray histogram only
    "GrayHist": {
        "description": "Grayscale intensity histogram only.",
        "feature_names": ["gray_hist"],
    },

    # 3) color histograms (RGB + HSV)
    "ColorHists": {
        "description": "Color histograms in RGB and HSV.",
        "feature_names": ["rgb_hist", "hsv_hist"],
    },

    # 4) texture-based using LBP + GLCM
    "Texture_LBP_GLCM": {
        "description": "Texture features using LBP and GLCM.",
        "feature_names": ["lbp_hist", "glcm_texture"],
    },

    # 5) gradients + texture
    "EdgesAndTexture": {
        "description": "Gradient info (Sobel hist) + texture (LBP).",
        "feature_names": ["sobel_hist", "lbp_hist"],
    },

    # 6) frequency-based
    "Frequency_FFT_DCT": {
        "description": "Frequency features using FFT radial + DCT low block.",
        "feature_names": ["fft_radial", "dct_low_block"],
    },

    # 7) morphology + intensity
    "Morphology_GrayHist": {
        "description": "Shape features from morphology + gray histogram.",
        "feature_names": ["shape_morphology", "gray_hist"],
    },

    # 8) a richer combined pipeline
    "Combo_BestGuess": {
        "description": "Combination of gray hist, LBP, Sobel, FFT radial.",
        "feature_names": ["gray_hist", "lbp_hist", "sobel_hist", "fft_radial"],
    },
}
