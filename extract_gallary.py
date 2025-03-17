import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


from sample4geo.model import TimmModel

class Config:
    model_path = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384'
    img_size = 384
    batch_size = 128
    gpu_ids = (0,)
    normalize_features = True
    gallery_folder = '/home/gpu/Desktop/Data/campus_data_with_indicies/gallery_satellite'
    checkpoint = '/home/gpu/Desktop/Sample4Geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4

def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
                
    return data     
class U1652DatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()
 

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())
                
        self.transforms = transforms
        
        self.given_sample_ids = sample_ids
        
        self.images = []
        self.sample_ids = []
        
        self.mode = mode
        
        
        self.gallery_n = gallery_n
        

        for i, sample_id in enumerate(self.ids):
                
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                    
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                      file))
                
                self.sample_ids.append(sample_id) 
  
    def __getitem__(self, index):
        
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1
        
        return img, label

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)
     
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
