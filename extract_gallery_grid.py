import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sample4geo.model import TimmModel
import re

class Config:
    model_path = './university_main/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    batch_size = 128
    gpu_ids = (0,)
    normalize_features = True
    gallery_folder = './Data/hall10_data/hall10_satellite_photos/gallery_satellite'
    checkpoint = './university_main/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    csv_path = './Data/hall10_data/gallery_satellite_image_coordinates.csv'
    output_file = 'gallery_data.npz'

def load_coordinates(csv_path):
    """Load image coordinates and assign correct (column, row) grid indices."""
    df = pd.read_csv(csv_path)

    def parse_filename(filename):
        """Extract Waypoint ID (row index) from filename."""
        match = re.match(r'path\d+_waypoint_(\d+)\.png', filename)
        if match:
            row_id = int(match.group(1))  # Extract row index from filename
            return row_id
        return None

    df['Row ID'] = df['Image Filename'].apply(parse_filename)
    
    # Extract Path ID directly from the last column of the CSV
    df['Path ID'] = df['Path ID'].astype(int)  # Ensure it's an integer

    # Convert to dictionary with (path_id, row_id) indexing
    coord_dict = {(row['Path ID'], row['Row ID']): 
                  (row['Latitude'], row['Longitude'], row['Image Filename']) 
                  for _, row in df.iterrows() if row['Row ID'] is not None}
    
    return coord_dict

class U1652DatasetEval(Dataset):
    def __init__(self, data_folder, transforms=None):
        super().__init__()
        self.image_paths = [os.path.join(root, file)
                            for root, _, files in os.walk(data_folder)
                            for file in files if file.endswith(('.png', '.jpg'))]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, img_path

    def __len__(self):
        return len(self.image_paths)

def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def extract_gallery_features():
    print("Loading model...")
    model = TimmModel(Config.model_path, pretrained=True, img_size=Config.img_size)
    model.load_state_dict(torch.load(Config.checkpoint), strict=False)
    model.to(Config.device)
    model.eval()
    
    print("Loading coordinates and dataset...")
    coord_dict = load_coordinates(Config.csv_path)
    val_transforms = get_transforms(Config.img_size)
    dataset = U1652DatasetEval(Config.gallery_folder, transforms=val_transforms)
    loader = DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False, pin_memory=True)
    
    grid_data = {}  # Dictionary to store features in grid format (columns=path_ids, rows=waypoints)

    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(Config.device)
            features = model(imgs)
            if Config.normalize_features:
                features = normalize(features, dim=-1)
            features = features.cpu().numpy()

            for img_path, feature in zip(img_paths, features):
                img_name = os.path.basename(img_path)

                # Find corresponding (path_id, row_id) grid position
                matching_keys = [key for key, val in coord_dict.items() if val[2] == img_name]
                if not matching_keys:
                    continue  # Skip if no match found

                path_id, row_id = matching_keys[0]
                lat, lon, _ = coord_dict[(path_id, row_id)]

                # Store in grid (grid_data[path_id][row_id] = {features, lat, lon})
                if path_id not in grid_data:
                    grid_data[path_id] = {}
                grid_data[path_id][row_id] = {
                    "features": feature,
                    "image_name": img_name,
                    "latitude": lat,
                    "longitude": lon
                }
    
    # Save grid data
    np.savez(Config.output_file, grid_data=grid_data)
    print(f"Saved grid-structured features to {Config.output_file}")

if __name__ == '__main__':
    extract_gallery_features()
