import numpy as np
from scipy import signal
from skimage.measure import label, regionprops

# from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu


def lbp_to_binary(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    threshold = threshold_otsu(image)
    return lbp_normalized <= threshold


# def calculate_centroid_distances(centroids_true, centroids_pred):
#     if len(centroids_true) == 0 or len(centroids_pred) == 0:
#         return 0
#     distances = np.min(
#         np.sqrt(((centroids_true[:, np.newaxis] - centroids_pred) ** 2).sum(axis=2)),
#         axis=1,
#     )
#     return np.mean(distances)


def calculate_centroid_distances(centroids_true, centroids_pred):
    if len(centroids_true) == 0 or len(centroids_pred) == 0:
        return 0
    distances = np.min(
        np.linalg.norm(centroids_true[:, np.newaxis] - centroids_pred, axis=2), axis=1
    )
    return np.mean(distances)


def calculate_mccorr(y_true, y_pred):
    # Convert to grayscale if the inputs are RGB
    if y_true.ndim == 3:
        y_true = rgb2gray(y_true)
    if y_pred.ndim == 3:
        y_pred = rgb2gray(y_pred)

    y_true_binary = lbp_to_binary(y_true)
    y_pred_binary = lbp_to_binary(y_pred)

    ccorr = signal.correlate2d(
        y_true_binary.astype(float), y_pred_binary.astype(float), mode="same"
    )
    ccorr_max = np.max(ccorr)

    props_true = regionprops(label(y_true_binary))
    props_pred = regionprops(label(y_pred_binary))

    centroids_true = np.array([prop.centroid for prop in props_true])
    centroids_pred = np.array([prop.centroid for prop in props_pred])

    avg_distance = calculate_centroid_distances(centroids_true, centroids_pred)

    mccorr = ccorr_max / (np.log2(avg_distance + 2))

    return mccorr


# # Example usage
# if __name__ == "__main__":
#     y_true_path = r"pytorch-CycleGAN-and-pix2pix\results\skeletonizer\test_latest\images\19MRTUBES3001TO3999CAM4_T131_L045_2019.07.31_090140_001_BI_real_B.png"  # Replace with your ground truth image path
#     y_pred_path = r"pytorch-CycleGAN-and-pix2pix\results\skeletonizer\test_latest\images\19MRTUBES3001TO3999CAM4_T131_L045_2019.07.31_090140_001_BI_fake_B.png"

#     mccorr_score = calculate_mccorr(y_true_path, y_pred_path)
#     print(f"\nFinal M-CCORR score: {mccorr_score:.4f}")
