import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sample4geo.dataset.university import U1652DatasetEval

from sample4geo.model import TimmModel

class Config:
    model_path = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    batch_size = 128
    gpu_ids = (0,)
    normalize_features = True
    gallery_folder = '/home/gpu/Desktop/Data/campus_data_with_indicies_single/gallery_satellite'
    checkpoint = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
   
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    
    return val_transforms
def extract_gallery_features():
    print("Loading model...")
    model = TimmModel(Config.model_path, pretrained=True, img_size=Config.img_size)
    model.load_state_dict(torch.load(Config.checkpoint), strict=False)
    model.to(Config.device)
    model.eval()
    
    print("Loading gallery dataset...")
    val_transforms = get_transforms((Config.img_size, Config.img_size))
    gallery_dataset = U1652DatasetEval(Config.gallery_folder, mode="gallery", transforms=val_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False, pin_memory=True)
    
    features_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(gallery_loader, desc="Extracting gallery features"):
            imgs = imgs.to(Config.device)
            features = model(imgs)
            if Config.normalize_features:
                features = normalize(features, dim=-1)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    gallery_f = np.concatenate(features_list, axis=0)
    gallery_label = np.concatenate(labels_list, axis=0)
    
    np.save('gallery_features.npy', gallery_f)
    np.save('gallery_labels.npy', gallery_label)
    print("Gallery features saved to gallery_features.npy and gallery_labels.npy")

if __name__ == '__main__':
    extract_gallery_features()
