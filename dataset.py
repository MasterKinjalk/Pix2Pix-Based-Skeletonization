import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import config


class SkeletonDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_files)

    def high_pass_sharpen(self, image):
        lowpass = cv2.GaussianBlur(image, (3, 3), 2)
        highpass = image - lowpass
        sharpened = image + highpass
        return np.uint8(np.clip(sharpened, 0, 255))

    def clahe_color(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
        equalized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        return equalized_image

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        image = cv2.imread(img_path)

        if image is None:
            raise IOError(f"Failed to load image at {img_path}")

        h, w, _ = image.shape
        root_image = image[:, : w // 2, :]
        skeleton_image = image[:, w // 2 :, :]

        root_normalized = self.clahe_color(root_image)
        root_sharpened = self.high_pass_sharpen(root_normalized)
        skeleton_sharpened = self.high_pass_sharpen(skeleton_image)

        try:
            # Apply transforms from config
            root_tensor = config.transform_only_input(image=root_sharpened)["image"]
            skeleton_tensor = config.transform_only_mask(image=skeleton_sharpened)[
                "image"
            ]
        except Exception as e:
            print(f"Error applying transforms to image {img_path}: {str(e)}")
            raise

        return root_tensor, skeleton_tensor
