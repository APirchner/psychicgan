"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageChops
from torchvision import transforms

class VideoData(data.Dataset):

    def __init__(self,block_in,block_out,overlap,folder_path):
        self.root_dir = folder_path                   # check out how to do this elegantly
        self.block_in = block_in
        self.block_out = block_out
        self.block_overlap = overlap
        
        self.all_trials = sorted([os.path.join(self.root_dir, name) for name in 
                                   os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, name))])
        self.all_trials = [trial+'/image_02/data' for trial in self.all_trials]
        self.all_frames = [len(os.listdir(imagedir)) for imagedir in self.all_trials]
        self.all_lens = [(nframes-self.block_overlap)//(self.block_in+self.block_out-self.block_overlap) 
                         for nframes in self.all_frames]
        
    def __getitem__(self,key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")
            
    def __len__(self):
        return np.sum(self.all_lens)
        
    def get_item_from_index(self, index):
        folder_index = 0
        while index>self.all_lens[folder_index]:
            index-=self.all_lens[folder_index]
            folder_index+=1
            
        folder_to_read = self.all_trials[folder_index]
        all_images = sorted(os.listdir(folder_to_read))
        start_idx = index*(self.block_in+self.block_out-self.block_overlap)
        images_past = all_images[start_idx:start_idx+self.block_in]
        images_future = all_images[start_idx+self.block_in:start_idx+self.block_in+self.block_out]
        
        to_tensor = transforms.ToTensor()

        imgs = [Image.open(os.path.join(folder_to_read,x)) for x in images_past]
        imgs = torch.stack([to_tensor(img) for img in imgs],dim=1)

        targets = [Image.open(os.path.join(folder_to_read,x)) for x in images_future]
        targets = torch.stack([to_tensor(img) for img in targets],dim=1)

        return imgs, targets