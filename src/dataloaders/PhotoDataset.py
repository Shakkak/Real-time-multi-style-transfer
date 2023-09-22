import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T
import glob
class PhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = root_dir
        self.transform = transform
        self.list_images = glob.glob(os.path.join(self.root_dir,'*.jpg'))
        
    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self,idx):
        img_path = self.list_images[idx]
        img = read_image(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        
        return img, img_path.split('/')[-1].split('.')[0]
        
 