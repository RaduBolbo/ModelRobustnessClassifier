import os, torch, random, shutil, numpy as np, pandas as pd
from glob import glob; from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
torch.manual_seed(2024)
import cv2


class CustomDataset(Dataset):
    
    def __init__(self, root, transformations = None):
   
        self.transformations = transformations
        self.im_paths = glob(f"{root}/*/*.jpeg")        
        self.cls_names, self.cls_counts, count = {}, {}, 0
            
        for idx, im_path in enumerate(self.im_paths):
            cls_name = self.get_cls_name(im_path)
            if cls_name not in self.cls_names: self.cls_names[cls_name] = count; count += 1
            if cls_name not in self.cls_counts: self.cls_counts[cls_name] = 1
            else: self.cls_counts[cls_name] += 1
    
        
    def get_cls_name(self, path): return os.path.dirname(path).split("/")[-1]
    
    def __len__(self): return len(self.im_paths)

    def get_pos_neg_im_paths(self, qry_label):
        
        pos_im_paths = [im_path for im_path in self.im_paths if qry_label == self.get_cls_name(im_path)]
        neg_im_paths = [im_path for im_path in self.im_paths if qry_label != self.get_cls_name(im_path)]
        
        pos_rand_int = random.randint(a = 0, b = len(pos_im_paths) - 1)
        neg_rand_int = random.randint(a = 0, b = len(neg_im_paths) - 1)
        
        return pos_im_paths[pos_rand_int], neg_im_paths[neg_rand_int]
    
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        qry_im = Image.open(im_path).convert("RGB")
        qry_label = self.get_cls_name(im_path)

        pos_im_path, neg_im_path = self.get_pos_neg_im_paths(qry_label = qry_label)
        pos_im, neg_im = Image.open(pos_im_path).convert("RGB"), Image.open(neg_im_path).convert("RGB")

        qry_gt = self.cls_names[qry_label]
        neg_gt = self.cls_names[self.get_cls_name(neg_im_path)]

        if self.transformations is not None: qry_im = self.transformations(qry_im); pos_im = self.transformations(pos_im); neg_im = self.transformations(neg_im)

        data = {}

        data["qry_im"] = qry_im
        data["qry_gt"] = qry_gt
        data["pos_im"] = pos_im # **** not used for the moment. For normal trining only query is used
        data["neg_im"] = neg_im # **** not used for the moment. For normal trining only query is used
        data["neg_gt"] = neg_gt # **** not used for the moment. For normal trining only query is used
            
        return data
    
def get_dls(root, train_transformations, val_transformations, batch_size, split = [0.8, 0.2], num_workers = 4):
    
    ds = CustomDataset(root = root)
    
    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = total_len - tr_len
    
    tr_ds, vl_ds  = random_split(dataset = ds, lengths = [tr_len, vl_len])

    # add the propper transformations
    tr_ds.dataset.transformations = train_transformations
    vl_ds.dataset.transformations = val_transformations

    tr_dl, val_dl = DataLoader(tr_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers), DataLoader(vl_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    return tr_dl, val_dl, ds.cls_names, ds.cls_counts

def display_tensor(tensor):
    print(tensor.shape)
    array = tensor.cpu().numpy()
    print(np.min(array), np.max(array))
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) # normalize
    array = np.transpose(array, (1, 2, 0))
    print(np.min(array), np.max(array))

    cv2.imshow('Image', array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root = "dataset/raw-img"
    mean, std, size, batch_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224, 16 # media, deviatai standard, diemsniuenaimaginilor si batch size-ul
    tfs = T.Compose([T.ToTensor(), T.Resize(size = (size, size), antialias = False), T.Normalize(mean = mean, std = std)])
    tr_dl, val_dl, classes, cls_counts = get_dls(root = root, transformations = tfs, batch_size = batch_size)

    print(len(tr_dl))
    print(len(val_dl))
    print(classes)
    print(cls_counts)

    for batch in tr_dl:
        print(batch['qry_im'].shape)
        print(batch['qry_gt'].shape)
        print(batch['pos_im'].shape)
        print(batch['neg_im'].shape)
        print(batch['neg_gt'].shape)
        print(batch['qry_gt'])
        print(batch['neg_gt'])
        
        display_tensor(batch['qry_im'][0, :, :, :])
        display_tensor(batch['pos_im'][0, :, :, :])
        display_tensor(batch['neg_im'][0, :, :, :])
        break
