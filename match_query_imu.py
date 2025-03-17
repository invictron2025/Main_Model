import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.functional import normalize
from geopy.distance import geodesic
from sample4geo.model import TimmModel
from imu_utils import estimate_position  # Function to estimate position from IMU data

class Config:
    model_path = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    gpu_ids = (0,)
    normalize_features = True
    query_folder = '/home/gpu/Desktop/Data/hall_10_query_photos/query_drone/0'
    checkpoint = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gallery_data_file = 'gallery_data.npz'  # Using grid structure
   
    max_search_radius = 3  # Search within a grid radius of 3 path/row IDs

def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def load_gallery_data():
    """Load gallery features and structured grid format."""
    data = np.load(Config.gallery_data_file, allow_pickle=True)['grid_data'].item()
    return data  # Structured as {path_id: {row_id: {features, image_name, lat, lon}}}

def find_nearest_grid(estimated_position, gallery_grid):
    """Find nearest Path ID and Row ID in the stored grid based on estimated GPS coordinates."""
    min_dist = float('inf')
    nearest_path_id, nearest_row_id = None, None

    est_lat, est_lon = estimated_position

    for path_id in gallery_grid:
        for row_id in gallery_grid[path_id]:
            lat, lon = gallery_grid[path_id][row_id]['latitude'], gallery_grid[path_id][row_id]['longitude']
            distance = geodesic((est_lat, est_lon), (lat, lon)).meters

            if distance < min_dist:
                min_dist = distance
                nearest_path_id, nearest_row_id = path_id, row_id

    return nearest_path_id, nearest_row_id

def get_nearby_gallery_features(gallery_grid, nearest_path_id, nearest_row_id):
    """Get nearby satellite images from the grid centered on (nearest_path_id, nearest_row_id)."""
    nearby_features = []
    metadata = []

    for path_id in range(nearest_path_id - Config.max_search_radius, nearest_path_id + Config.max_search_radius + 1):
        if path_id not in gallery_grid:
            continue
        for row_id in range(nearest_row_id - Config.max_search_radius, nearest_row_id + Config.max_search_radius + 1):
            if row_id not in gallery_grid[path_id]:
                continue

            entry = gallery_grid[path_id][row_id]
            nearby_features.append(entry['features'])
            metadata.append(entry)

    return np.array(nearby_features), metadata

def find_best_match(query_feature, nearby_features, metadata):
    """Finds the best match within the nearby grid search space."""
    if len(nearby_features) == 0:
        print("No nearby matches found within grid window. Expanding search...")
        return None

    scores = np.einsum('ij,j->i', nearby_features, query_feature)
    best_match_idx = np.argmax(scores)
    return metadata[best_match_idx]

def load_model():
    jit_path = Config.model_path + "_jit.pt"
    if os.path.exists(jit_path):
        model = torch.jit.load(jit_path, map_location=Config.device)
    else:
        model = TimmModel(Config.model_path, pretrained=False, img_size=Config.img_size)
        model.load_state_dict(torch.load(Config.checkpoint, map_location=Config.device), strict=False)
        model.to(Config.device)
        model.eval()
        example_input = torch.randn(1, 3, Config.img_size, Config.img_size).to(Config.device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(jit_path)
    return model

def process_single_query(model, gallery_grid, last_known_position, velocity, acceleration, heading,dt):
    """Estimate position using IMU, search nearby grid, and find the best match."""
    img_path = next((os.path.join(Config.query_folder, f) for f in os.listdir(Config.query_folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = get_transforms(Config.img_size)
    img = transforms(image=img)['image'].unsqueeze(0).to(Config.device)

    # Extract query feature
    with torch.no_grad():
        query_feature = model(img)
        if Config.normalize_features:
            query_feature = normalize(query_feature, dim=-1)
        query_feature = query_feature.cpu().numpy().squeeze()

    # Estimate new position using IMU data
    estimated_position = estimate_position(last_known_position[0], last_known_position[1], velocity, acceleration, heading, dt)


    # Find nearest grid location
    nearest_path_id, nearest_row_id = find_nearest_grid(estimated_position, gallery_grid)
    print(f"Nearest Grid Cell: Path {nearest_path_id}, Row {nearest_row_id}")

    # Get nearby gallery images from structured grid
    nearby_features, metadata = get_nearby_gallery_features(gallery_grid, nearest_path_id, nearest_row_id)

    total_images_in_search_area = len(metadata)
    print(f"Total images in nearby search area: {total_images_in_search_area}")

    # Find best match within the grid search space
    best_match = find_best_match(query_feature, nearby_features, metadata)

    if best_match is None:
        print(f"No match found within the search grid window. Try expanding the search.")
    else:
        print(f"Query Image: {os.path.basename(img_path)}, Match: {best_match['image_name']}, "
              f"Estimated Position: {estimated_position}, Match Coordinates: ({best_match['latitude']}, {best_match['longitude']})")

    return estimated_position

if __name__ == '__main__':
    model = load_model()
    gallery_grid = load_gallery_data()
    last_known_position = (26.5115960437853,80.22629763462655)  # Initial starting coordinates
    velocity, acceleration, heading = (1, 1, 3.14)  # Replace with actual IMU data
    dt = 1
    process_single_query(model, gallery_grid, last_known_position, velocity, acceleration, heading,dt)
