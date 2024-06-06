from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import h5py
import os
from utils.transforms import transform_sample
import glob
from torchvision.transforms import functional
import torchvision.transforms as T
import torch

class crowd_dataset(Dataset):
    def __init__(self, dataset_dir, dataset_type, image_size = 384, is_train = False):
        self.is_train = is_train
        self.image_size = image_size
        self.dataset_type  = dataset_type
        if is_train:
            is_train = 'train_data'
        else:
            is_train = 'valid_data'
        if dataset_type == 'SHA':
            dataset_type = os.path.join('shanghaitech', 'part_A')
        elif dataset_type == 'SHB':
            dataset_type = os.path.join('shanghaitech', 'part_B')
        #            
        self.trans      = transform_sample((-2, 2), (0.8, 1.2), (image_size, image_size), 2, (0.5, 1.5), self.dataset_type)
        self.image_list = glob.glob(os.path.join(dataset_dir, dataset_type, is_train, 'images', '*.jpg'))
        #        
    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')		
        label = h5py.File(self.image_list[index].replace('images', 'density_map').replace('.jpg', '.h5'), 'r')
        gt    = np.array(label['gt'], dtype = np.float32)
        #
        if self.is_train:
            density_hr = np.array(label['density_hr'], dtype = np.float32)
            density_mr = np.array(label['density_mr'], dtype = np.float32)
            density_lr = np.array(label['density_lr'], dtype = np.float32)
            attention  = np.array(label['attention'])
            image, density_hr, density_mr, density_lr, attention = self.trans(image, density_hr, density_mr, density_lr, attention)						
            density_hr = torch.from_numpy(density_hr)
            density_mr = torch.from_numpy(density_mr)
            density_lr = torch.from_numpy(density_lr)
            attention  = torch.from_numpy(attention)	
            image      = T.ToTensor()(image)
            image      = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            count      = density_hr.sum()
            return image, density_hr, density_mr, density_lr, attention, count
        else:
            height, width = image.size[1], image.size[0]
            height        = round(height / 32) * 32
            width         = round(width  / 32) * 32
            image         = image.resize((width, height), Image.BILINEAR)
            image         = T.ToTensor()(image)
            image         = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image, gt

    def __len__(self):
        return len(self.image_list)
