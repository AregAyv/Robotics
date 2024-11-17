import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import cv2
import numpy as np


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


# Function to process an image and predict density map
def predict_density_map(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]

    # Preprocess the image for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_resized = cv2.resize(image, (1024, 768))  # Resize for uniform input
    image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension

    # Pass through the model
    with torch.no_grad():
        density_map = model(image_tensor)
    density_map = density_map.squeeze().cpu().numpy()

    # Resize density map back to original image size
    density_map_resized = cv2.resize(density_map, (original_shape[1], original_shape[0]))

    return density_map_resized


# Function to count people from the density map
def count_people(density_map):
    return np.sum(density_map)


# Main script
if __name__ == "__main__":
    # Initialize the model
    model = CSRNet()

    # Load the pre-trained weights
    weights_path = "D:\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final"  # Replace with the correct path to the weights file
    try:
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )
        print(f"Successfully loaded weights from {weights_path}.")
    except FileNotFoundError:
        print(f"Error: Weights file not found at '{weights_path}'. Please ensure it exists and is accessible.")
        exit(1)

    model.eval()

    # Path to the image
    image_path = "D:\human.jpg"  # Replace with the correct path to your image

    # Predict density map
    density_map = predict_density_map(image_path, model)

    # Count people
    people_count = count_people(density_map)

    # Display results
    print(f"Estimated People Count: {int(people_count)}")

    # Normalize density map for visualization
    density_map_normalized = (density_map / density_map.max() * 255).astype(np.uint8)
    density_map_colored = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)

    # Save the result as an image file for visualization
    cv2.imwrite("density_map_output.jpg", density_map_colored)
    print("Density map saved as 'density_map_output.jpg'.")
