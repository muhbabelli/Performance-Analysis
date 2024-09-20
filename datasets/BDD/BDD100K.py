import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class BDD100KDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])

        assert len(self.image_files) == len(self.label_files), "Number of images and labels must be the same."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        image = np.array(Image.open(image_path).convert('RGB'))
        label = np.array(Image.open(label_path))

        if self.transform:
            aug = self.transform(image=image, mask=label)
            image, label = aug["image"], aug["mask"]

        return image, label

    def get_image_path(self, idx):
        return os.path.join(self.images_dir, self.image_files[idx])

    def get_label_path(self, idx):
        return os.path.join(self.labels_dir, self.label_files[idx])

    def get_image_paths(self):
        image_paths= []
        for img in self.image_files:
            image_paths.extend([os.path.join(self.images_dir, img)])
        return image_paths
    
    def get_image_names(self):
        return self.image_files

# Example usage
if __name__ == "__main__":
    from torchvision import transforms
    from tqdm import tqdm

    # Define a transform function using torchvision transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    images_path = "/path/to/val/images"
    labels_path = "/path/to/val/labels"
    
    bdd_dataset = BDD100KDataset(images_path, labels_path, transform)
    data_loader = DataLoader(bdd_dataset, batch_size=1, shuffle=False, num_workers=15)

    for idx, elm in enumerate(tqdm(data_loader)):
        if idx == 10:
            break
        img, smnt = elm
        print(f"Image shape: {img.shape}, Label shape: {smnt.shape}")

        # Access the image and label paths
        img_path = bdd_dataset.get_image_path(idx)
        label_path = bdd_dataset.get_label_path(idx)
        print(f"Image path: {img_path}, Label path: {label_path}")
