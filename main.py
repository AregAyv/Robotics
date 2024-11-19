import os
import numpy as np
import scipy.io
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm


# Define the CSRNet model
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        # Use the frontend from pre-trained VGG16
        self.frontend = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23]

        # Define the backend layers
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


# Function to generate density map from points
def gaussian_filter_density(img, points):
    density = np.zeros(img.shape[:2], dtype=np.float32)
    h, w = density.shape

    if len(points) == 0:
        return density

    tree = KDTree(points, leafsize=2048)
    distances, _ = tree.query(points, k=4)

    for i, point in enumerate(points):
        x, y = min(w - 1, max(0, int(point[0]))), min(h - 1, max(0, int(point[1])))

        if len(points) > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) / 3.0
        else:
            sigma = np.mean([h, w]) / 4.0

        gaussian_map = np.zeros_like(density)
        gaussian_map[y, x] = 1
        density += gaussian_filter(gaussian_map, sigma, mode='constant')

    return density


# Function to generate density maps for dataset
def generate_density_maps(root_dir):
    img_dir = os.path.join(root_dir, "images")
    gt_dir = os.path.join(root_dir, "ground_truth")
    density_dir = os.path.join(root_dir, "density_maps")

    os.makedirs(density_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(img_dir), desc="Generating density maps"):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(img_dir, img_file)
            gt_path = os.path.join(gt_dir, f"GT_{os.path.splitext(img_file)[0]}.mat")
            density_path = os.path.join(density_dir, f"{os.path.splitext(img_file)[0]}.npy")

            img = Image.open(img_path)
            img_array = np.array(img)

            mat = scipy.io.loadmat(gt_path)
            points = mat["image_info"][0, 0][0, 0][0]

            density_map = gaussian_filter_density(img_array, points)
            np.save(density_path, density_map)


# Custom Dataset Class
class ShanghaiTechDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.density_dir = os.path.join(root_dir, "density_maps")
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        density_path = os.path.join(self.density_dir, f"{os.path.splitext(img_file)[0]}.npy")

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Load density map
        density_map = np.load(density_path)
        density_map = torch.from_numpy(density_map).unsqueeze(0).float()  # Add channel dimension

        return img, density_map


# Training Function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, density_maps in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            density_maps = density_maps.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, density_maps)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    dataset_root = "D:\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data"  # Update path if needed

    # Generate density maps
    print("Generating density maps for dataset...")
    generate_density_maps(dataset_root)
    print("Density maps generation completed.")

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ShanghaiTechDataset(dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize the CSRNet model
    model = CSRNet()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Train the model
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)

    # Save the trained model
    weights_path = "csrnet_trained_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}.")
