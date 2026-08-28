
import cv2
import numpy as np

# Fixed dimension for standardizing input vectors
IMAGE_SIZE = (128, 128)

def load_and_preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocesses a single image for feature extraction:
    1. Reads image from path.
    2. Resizes to fixed target dimensions.
    3. Converts to BGR->RGB (for Color Histogram).
    4. Converts to Grayscale (for HOG and LBP).
    5. Applies Gaussian Blur for light noise reduction.
    
    Returns:
        tuple: (processed_gray_img, processed_rgb_img) or (None, None) if corrupt.
    """
    # 1. Read image from disk
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None or img_bgr.size == 0:
        return None, None

    # 2. Resize to fixed target dimension
    resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    # 3. Convert to RGB for Color Histogram
    img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # 4. Convert to Grayscale for HOG & LBP
    img_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # 5. Noise reduction using 3x3 Gaussian Blur
    img_gray_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

    return img_gray_blurred, img_rgb

if __name__ == "__main__":
    # Test run on a sample image from the dataset
    import os
    sample_img_path = os.path.join("dataset", "buffalo", "banni")
    
    if os.path.exists(sample_img_path):
        sample_files = os.listdir(sample_img_path)
        if sample_files:
            test_file = os.path.join(sample_img_path, sample_files[0])
            gray, rgb = load_and_preprocess_image(test_file)
            if gray is not None:
                print(f"✅ Preprocessing functional!")
                print(f"Grayscale image shape : {gray.shape}")
                print(f"RGB image shape       : {rgb.shape}")
                