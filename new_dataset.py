import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2


class SkeletonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_files)

    def sharpen_image(self, image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def non_max_suppression(self, image):
        # Compute gradients
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude and direction
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * (180 / np.pi) % 180

        # Quantize the angle
        angle_quantized = np.round(angle / 45) % 4

        # Non-maximum suppression
        nms = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                q = 255
                r = 255

                # Horizontal edge
                if (0 <= angle_quantized[i, j] < 1) or (
                    3 <= angle_quantized[i, j] <= 4
                ):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Vertical edge
                elif 1 <= angle_quantized[i, j] < 2:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # +45 degree edge
                elif 2 <= angle_quantized[i, j] < 3:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # -45 degree edge
                elif 3 <= angle_quantized[i, j] <= 4:
                    q = magnitude[i + 1, j + 1]
                    r = magnitude[i - 1, j - 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    nms[i, j] = magnitude[i, j]

        # Thresholding
        threshold = 20  # You may need to adjust this value
        nms[nms < threshold] = 0
        nms[nms >= threshold] = 255

        return nms.astype(np.uint8)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        image = cv2.imread(img_path)

        # Split the image into root and skeleton
        h, w, _ = image.shape
        root_image = image[:, : w // 2, :]
        skeleton_image = image[:, w // 2 :, :]

        # Convert to grayscale
        root_gray = cv2.cvtColor(root_image, cv2.COLOR_BGR2GRAY)
        skeleton_gray = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2GRAY)

        # Apply histogram normalization to root image
        root_normalized = cv2.equalizeHist(root_gray)

        # Sharpen root image
        root_sharpened = self.sharpen_image(root_normalized)

        # Apply NMS to skeleton image
        skeleton_thinned = self.non_max_suppression(skeleton_gray)

        # Convert back to PIL Images
        root_pil = Image.fromarray(root_sharpened)
        skeleton_pil = Image.fromarray(skeleton_thinned)

        if self.transform:
            root_pil = self.transform(root_pil)
            skeleton_pil = self.transform(skeleton_pil)

        return root_pil, skeleton_pil


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

if __name__ == "__main__":
    dataset = SkeletonDataset("Data_Split/train/", transform=transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    for x, y in loader:
        print(f"Root image shape: {x.shape}, Skeleton image shape: {y.shape}")
        save_image(x, "root_sample.png")
        save_image(y, "skeleton_sample.png")
        break  # Just process one batch for this example
