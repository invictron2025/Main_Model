import torch
import numpy as np
from tqdm import tqdm
import gc
import time
import torch.nn.functional as F
# from ..trainer import predict
import scipy.io
from torch.amp import autocast

def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    
    
    print("Extracting Features:")
    
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

def predict(config, model, dataloader):
    device_type = "cuda" if "cuda" in str(config.device) else "cpu"
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast(device_type=device_type):
         
                img = img.to(config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(config.device)
        
    if config.verbose:
        bar.close()
        
    return img_features, ids_list

def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    
    
    print("Extracting Features:")
    
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
    # Convert to NumPy
    query_f = img_features_query.cpu().numpy()
    query_label = ids_query.cpu().numpy()
    gallery_f = img_features_gallery.cpu().numpy()
    gallery_label = ids_gallery.cpu().numpy()
    
   
    
    # Save results in .mat file
    result = {
        'query_f': query_f,
        'query_label': query_label,
        'gallery_f': gallery_f,
        'gallery_label': gallery_label,
        
    }
    
    scipy.io.savemat('pytorch_result.mat', result)
    print("Results saved to pytorch_result.mat")
    
    print("Compute Scores:")
    cmc_top1_indices = []
    cmc_top1_labels = []
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], query_label[i], img_features_gallery, gallery_label)

        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        if CMC_tmp[0] == 1:
            cmc_top1_indices.append(i)
            cmc_top1_labels.append(query_label[i])  # Extract the label
    # Save indices, labels, and paths to a text file
    with open('cmc_top1_indices_labels_paths.txt', 'w') as f:
        for index, label in zip(cmc_top1_indices, cmc_top1_labels):
            f.write(f"{index}, {label}\n")
        
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    print(' - '.join(string)) 
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[0]


def eval_query(qf,ql,gf,gl):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()
 
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]    

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc




