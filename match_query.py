import torch
import numpy as np
import scipy.io
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.model import TimmModel

class Config:
    model_path = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    batch_size = 1  # Process one query at a time
    gpu_ids = (0,)
    normalize_features = True
    query_folder = '/home/gpu/Desktop/Data/campus_data_with_indicies/query_drone'
    checkpoint = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    gallery_features_file = 'gallery_features.mat'

def load_gallery_features():
    print("Loading gallery features...")
    data = scipy.io.loadmat(Config.gallery_features_file)
    gallery_features = data['gallery_f']
    gallery_labels = data['gallery_label'].flatten()  # Ensure labels are a 1D array
    print(f"Loaded {gallery_features.shape[0]} gallery features.")
    return gallery_features, gallery_labels

def find_best_match(query_feature, gallery_features, gallery_labels):
    scores = gallery_features @ query_feature.T  # Compute similarity scores
    top_match_idx = np.argmax(scores)
    if top_match_idx >= len(gallery_labels):
        print("Warning: top_match_idx out of bounds.")
        return -1, -1
    return gallery_labels[top_match_idx], scores[top_match_idx]

def process_queries():
    print("Loading model...")
    model = TimmModel(Config.model_path, pretrained=False, img_size=Config.img_size)  # Disable online loading
    model.load_state_dict(torch.load(Config.checkpoint, map_location=Config.device), strict=False)
    model.to(Config.device)
    model.eval()
    
    print("Loading query dataset...")
    val_transforms, _, _ = get_transforms((Config.img_size, Config.img_size))
    query_dataset = U1652DatasetEval(Config.query_folder, mode="query", transforms=val_transforms)
    query_loader = DataLoader(query_dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False, pin_memory=True)
    
    gallery_features, gallery_labels = load_gallery_features()
    
    with torch.no_grad():
        for img, label in tqdm(query_loader, desc="Processing queries"):
            img = img.to(Config.device)
            query_feature = model(img)
            if Config.normalize_features:
                query_feature = normalize(query_feature, dim=-1)
            query_feature = query_feature.cpu().numpy().squeeze()
            
            best_match_label, confidence = find_best_match(query_feature, gallery_features, gallery_labels)
            print(f"Query Label: {label.item()}, Best Match: {best_match_label}, Confidence: {confidence}")
            
            if best_match_label == -1:
                print("Error: No valid match found.")

if __name__ == '__main__':
    process_queries()
