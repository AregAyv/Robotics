import os
import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import h5py
import random
import torch.nn.functional as F


# Function to load image and ground truth data
def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    if True:  # Random cropping (currently disabled)
        crop_size = (img.size[0] / 2, img.size[1] / 2)
        if random.randint(0, 9) <= -1:
            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:  # Flip the image and target randomly
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Resize the target density map
    target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return img, target


# CSRNet Model Definition
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 128 * 128, 1024)  # Updated fc1 to match flattened size
        self.fc2 = nn.Linear(1024, 1)  # Final output layer

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))  # First convolution layer
        x = F.relu(self.conv2(x))  # Second convolution layer
        x = self.pool(x)  # Max pooling layer

        # Print the shape of the tensor before flattening to debug
        print(x.shape)

        # Flatten the tensor to pass it to the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Final output layer

        return x


# Function to generate density map using Gaussian filtering
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


# Function to generate density maps for all images in the folder
def generate_density_maps(image_folder, ground_truth_folder, output_folder, sigma=15):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(image_folder):
        if img_name.endswith('.jpg'):
            base_name = os.path.splitext(img_name)[0]
            gt_path = os.path.join(ground_truth_folder, f"GT_{base_name}.mat")

            if not os.path.exists(gt_path):
                print(f"Ground truth file not found for image {img_name}. Skipping...")
                continue

            gt_data = sio.loadmat(gt_path)
            points = gt_data['image_info'][0][0][0][0][0]

            img_path = os.path.join(image_folder, img_name)
            img = Image.open(img_path)
            width, height = img.size

            density_map = np.zeros((height, width), dtype=np.float32)
            for point in points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < width and 0 <= y < height:
                    density_map[y, x] += 1

            density_map = gaussian_filter(density_map, sigma=sigma)

            output_path = os.path.join(output_folder, f"{base_name}.npy")
            np.save(output_path, density_map)

            print(f"Saved density map for {img_name} to {output_path}.")


# Custom Dataset class for loading images and density maps
class CustomDataset(Dataset):
    def __init__(self, image_paths, density_paths, target_size=(1024, 768)):
        self.image_paths = image_paths
        self.density_paths = density_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        density_path = self.density_paths[idx]

        # Load the image and density map using the load_data function
        image, density_map = load_data(image_path)

        # Resize the image and density map
        image = image.resize(self.target_size)
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Convert to tensor and normalize

        # Resize the density map to the target size
        density_map = cv2.resize(density_map, self.target_size[::-1])
        density_map = torch.tensor(density_map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, density_map


# Function to prepare DataLoader for training
def prepare_dataloader(image_dir, density_dir, batch_size=4, target_size=(1024, 768)):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]
    density_paths = [os.path.join(density_dir, img.replace(".jpg", ".npy")) for img in os.listdir(image_dir) if
                     img.endswith(".jpg")]

    for image_path, density_path in zip(image_paths, density_paths):
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist.")
        if not os.path.exists(density_path):
            print(f"Warning: Density map file {density_path} does not exist.")

    dataset = CustomDataset(image_paths, density_paths, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


# Function to train the CSRNet model
def train_csrnet(model, dataloader, criterion, optimizer, device, epochs=10):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, density_maps in dataloader:
            images, density_maps = images.to(device), density_maps.to(device)

            optimizer.zero_grad()
            predicted_density_map = model(images)

            loss = criterion(predicted_density_map, density_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader)}')


# Main function to initialize the training process
if __name__ == "__main__":
    image_folder = "D://train_data//images"
    ground_truth_folder = "D://train_data//ground_truth"
    density_folder = "D://train_data//density_maps"

    # Generate density maps from the images and ground truth
    generate_density_maps(image_folder, ground_truth_folder, density_folder, sigma=15)

    # Prepare the dataloader
    dataloader = prepare_dataloader(image_folder, density_folder, batch_size=4, target_size=(1024, 768))

    # Initialize model, loss function, and optimizer
    model = CSRNet()
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = Adam(model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_csrnet(model, dataloader, criterion, optimizer, device, epochs=10)
