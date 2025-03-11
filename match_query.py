import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sample4geo.dataset.university import U1652DatasetEval
import cv2

from sample4geo.model import TimmModel

class Config:
    model_path = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    batch_size = 1  # Process one query at a time
    gpu_ids = (0,)
    normalize_features = True
    query_folder = '/home/gpu/Desktop/Data/campus_data_with_indicies_single/query_drone'
    checkpoint = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0
    gallery_features_file = 'gallery_features.npy'
    gallery_labels_file = 'gallery_labels.npy'
   
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    
    return val_transforms
def load_gallery_features():
    print("Loading gallery features...")
    gallery_features = np.load(Config.gallery_features_file)
    gallery_labels = np.load(Config.gallery_labels_file)
    print(f"Loaded {gallery_features.shape[0]} gallery features.")
    return gallery_features, gallery_labels

def find_top_k_matches(query_feature, gallery_features, gallery_labels, k=5):
    scores = gallery_features @ query_feature.T  # Compute similarity scores
    top_k_indices = np.argsort(scores)[-k:][::-1]  # Get top K indices in descending order
    
    top_k_labels = gallery_labels[top_k_indices]  
    top_k_scores = scores[top_k_indices]  

    return top_k_labels, top_k_scores

def process_queries():
    print("Loading model...")
    model = TimmModel(Config.model_path, pretrained=False, img_size=Config.img_size)  # Disable online loading
    model.load_state_dict(torch.load(Config.checkpoint, map_location=Config.device), strict=False)
    model.to(Config.device)
    model.eval()
    
    print("Loading query dataset...")
    val_transforms = get_transforms((Config.img_size, Config.img_size))
    query_dataset = U1652DatasetEval(Config.query_folder, mode="query", transforms=val_transforms)
    query_loader = DataLoader(query_dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False, pin_memory=True)
    
    gallery_features, gallery_labels = load_gallery_features()
    
    total_queries = 0
    correct_matches = 0
    
    with torch.no_grad():
        for img, label in tqdm(query_loader, desc="Processing queries"):
            total_queries += 1
            img = img.to(Config.device)
            query_feature = model(img)
            if Config.normalize_features:
                query_feature = normalize(query_feature, dim=-1)
            query_feature = query_feature.cpu().numpy().squeeze()
            
            top_k_labels, top_k_scores = find_top_k_matches(query_feature, gallery_features, gallery_labels, k=5)
            
            if label.item() in top_k_labels:
                correct_idx = np.where(top_k_labels == label.item())[0][0]
                correct_matches += 1
                print(f"Query Label: {label.item()}, Correct Match Found: {top_k_labels[correct_idx]}, Confidence: {top_k_scores[correct_idx]}")
            else:
                print(f"Query Label: {label.item()}, Not Found in Top K Matches")
    
    accuracy = (correct_matches / total_queries) * 100 if total_queries > 0 else 0
    print(f"Top-K Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    process_queries()
