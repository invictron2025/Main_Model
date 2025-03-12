import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.functional import normalize
from sample4geo.model import TimmModel

class Config:
    model_path = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    gpu_ids = (0,)
    normalize_features = True
    query_folder = '/home/gpu/Desktop/Data/AirStrip_data/query_drone/9'
    checkpoint = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gallery_features_file = 'gallery_features.npy'
    gallery_labels_file = 'gallery_labels.npy'
    csv_file = '/home/gpu/Desktop/Sample4Geo/drone_airstrip.csv'  # Path to CSV file containing coordinates


def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_gallery_features():
    """Loads gallery features using memory-mapped files for speed."""
    # print("Loading gallery features (memory-mapped)...")
    gallery_features = np.load(Config.gallery_features_file, mmap_mode='r')
    gallery_labels = np.load(Config.gallery_labels_file, mmap_mode='r')
    return gallery_features, gallery_labels


def load_coordinates():
    """Loads the coordinates CSV file."""
    # print("Loading coordinates from CSV...")
    df = pd.read_csv(Config.csv_file)
    return df.to_numpy()  # Convert to NumPy array for faster indexing


def find_top_k_matches(query_feature, gallery_features, gallery_labels, k=1):
    """Efficient similarity search using einsum for fast matmul."""
    scores = np.einsum('ij,j->i', gallery_features, query_feature)  # Faster than np.dot()
    top_k_indices = np.argpartition(scores, -k)[-k:]  # Optimized top-k selection
    return gallery_labels[top_k_indices]


def load_model():
    """Loads the model efficiently and applies optimizations."""
    # print("Loading model (optimized)...")

    # Use a precompiled TorchScript model if available
    jit_path = Config.model_path + "_jit.pt"
    if os.path.exists(jit_path):
        # print("Using TorchScript compiled model.")
        model = torch.jit.load(jit_path, map_location=Config.device)
    else:
        model = TimmModel(Config.model_path, pretrained=False, img_size=Config.img_size)
        model.load_state_dict(torch.load(Config.checkpoint, map_location=Config.device), strict=False)
        model.to(Config.device)
        model.eval()

        # Save a TorchScript version for future runs
        example_input = torch.randn(1, 3, Config.img_size, Config.img_size).to(Config.device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(jit_path)
        # print(f"Saved TorchScript model for future use: {jit_path}")

    return model


def process_single_query(model, gallery_features, gallery_labels, coordinates):
    """Loads and processes a single query image, returning coordinates."""
    # print("Processing query image...")

    # Find the image file
    img_path = next((os.path.join(Config.query_folder, f) for f in os.listdir(Config.query_folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)

    if not img_path:
        raise FileNotFoundError(f"No image found in {Config.query_folder}")

    # Load and preprocess the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Faster loading
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = get_transforms(Config.img_size)
    img = transforms(image=img)['image'].unsqueeze(0).to(Config.device)

    # Extract query feature
    with torch.no_grad():
        query_feature = model(img)
        if Config.normalize_features:
            query_feature = normalize(query_feature, dim=-1)
        query_feature = query_feature.cpu().numpy().squeeze()

    # Find best match
    top_k_labels = find_top_k_matches(query_feature, gallery_features, gallery_labels, k=1)
    best_match_label = top_k_labels[0]

    # Retrieve the coordinates of the matched label
    matched_coordinates = coordinates[int(best_match_label)]  # Ensure label corresponds to row index

    print(f"Query Image: {os.path.basename(img_path)}, First Place Match: {best_match_label}, Coordinates: {matched_coordinates}")

    return best_match_label, matched_coordinates


if __name__ == '__main__':
    model = load_model()
    gallery_features, gallery_labels = load_gallery_features()
    coordinates = load_coordinates()
    process_single_query(model, gallery_features, gallery_labels, coordinates)
