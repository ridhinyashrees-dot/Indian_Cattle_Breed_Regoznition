import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# HOG Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP Parameters
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform'

# Color Histogram Parameters
COLOR_BINS = 8

def extract_hog_features(gray_img):
    """Extracts HOG features capturing body shape and edges."""
    return hog(
        gray_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        transform_sqrt=True
    )

def extract_lbp_features(gray_img):
    """Extracts normalized LBP histogram capturing texture details."""
    lbp = local_binary_pattern(gray_img, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    # Uniform LBP produces P + 2 bins
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_color_histogram(rgb_img):
    """Extracts 3D RGB color histogram capturing shade distribution."""
    # Compute 3D color histogram across R, G, B channels
    hist = cv2.calcHist(
        [rgb_img], 
        [0, 1, 2], 
        None, 
        [COLOR_BINS, COLOR_BINS, COLOR_BINS], 
        [0, 256, 0, 256, 0, 256]
    )
    # Normalize vector to ensure scale invariance
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_combined_features(gray_img, rgb_img):
    """
    PHASE 6: Concatenates HOG + LBP + Color Histogram vectors into a single 1D vector.
    """
    hog_feat = extract_hog_features(gray_img)
    lbp_feat = extract_lbp_features(gray_img)
    color_feat = extract_color_histogram(rgb_img)

    # Concatenate features into a single array
    combined_vector = np.hstack([hog_feat, lbp_feat, color_feat])
    return combined_vector

if __name__ == "__main__":
    import os
    from src.preprocess import load_and_preprocess_image

    sample_path = os.path.join("dataset", "buffalo", "banni")
    if os.path.exists(sample_path):
        sample_files = os.listdir(sample_path)
        if sample_files:
            test_file = os.path.join(sample_path, sample_files[0])
            gray, rgb = load_and_preprocess_image(test_file)
            
            if gray is not None and rgb is not None:
                hog_f = extract_hog_features(gray)
                lbp_f = extract_lbp_features(gray)
                col_f = extract_color_histogram(rgb)
                combined = extract_combined_features(gray, rgb)

                print(f"✅ Feature Extraction Module Ready!")
                print(f"  - HOG Vector Length   : {len(hog_f)}")
                print(f"  - LBP Vector Length   : {len(lbp_f)}")
                print(f"  - Color Hist Length   : {len(col_f)}")
                print(f"  - Combined Vector Size: {len(combined)}")