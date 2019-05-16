"""Data utility functions."""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image, ImageChops, ImageStat
from torchvision import transforms

class KITTIData(data.Dataset):

    def __init__(self,block_in,block_out,overlap,folder_path):
        self.root_dir = folder_path                   # check out how to do this elegantly
        self.block_in = block_in
        self.block_out = block_out
        self.block_overlap = overlap
        
        self.all_trials = sorted([os.path.join(self.root_dir, name) for name in 
                                   os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, name))])
        
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
        return len(self.all_trials)
        
    def get_item_from_index(self, index):
        folder_to_read = self.all_trials[index]
        all_images = sorted(os.listdir(folder_to_read))
        images_past = all_images[0:self.block_in]
        images_future = all_images[self.block_in:self.block_in+self.block_out]
        
        to_tensor = transforms.ToTensor()

        imgs = [Image.open(os.path.join(folder_to_read,x)) for x in images_past]
        imgs = torch.stack([to_tensor(img) for img in imgs],dim=1)

        targets = [Image.open(os.path.join(folder_to_read,x)) for x in images_future]
        targets = torch.stack([to_tensor(img) for img in targets],dim=1)

        return imgs, targets
    
def transform_dataset(block_in,block_out,overlap,path_old):
    path_up = os.path.normpath(path_old + os.sep + os.pardir)
    path_new = os.path.join(path_up,"in_%d_out_%d_ol_%d" %(block_in,block_out,overlap))
    os.mkdir(path_new)
    
    to_tensor = transforms.ToTensor()
    
    all_trials = sorted([os.path.join(path_old, name) for name in 
                         os.listdir(path_old) if os.path.isdir(os.path.join(path_old, name))])
    all_trials = [trial+'/image_02/data' for trial in all_trials]
    sets = 0
    for folder_to_read in all_trials:
        all_images = sorted(os.listdir(folder_to_read))
        # read in all the images
        imgs = [Image.open(os.path.join(folder_to_read,x)) for x in all_images]
        # calculate diffs between images for movement detection
        img_diffs = [ImageChops.difference(a,b) for a,b in zip(imgs,imgs[1:])]
        # transform diffs to tensors for easier manipulation
        img_diff_tens = [to_tensor(x).mean(dim=0) for x in img_diffs]
        # sum the pixels over longer edge (shows where movement is)
        hists = [x.sum(dim=0) for x in img_diff_tens]
        locs = [x.argmax() for x in hists]
        # generate crop boxes
        boxes,idxs = generate_boxes(locs,img_diff_tens[0].shape[1],img_diff_tens[0].shape[0],block_in,block_out,overlap)
        for start,box in zip(idxs,boxes):
            new_set = os.path.join(path_new,'%05d' % sets)
            os.mkdir(new_set)
            sets+=1
            for idx in range(start,start+block_in+block_out):
                new_img = imgs[idx].crop(box).resize((64,64))
                new_img.save(os.path.join(new_set,'%02d.png' % idx))
        
def generate_boxes(locs,leng,heig,b_in,b_out,b_ol):
    boxes = []
    idxs = []
    b_top = 0
    b_bottom = heig
    idx = 0
    while idx+b_in+b_out-b_ol<len(locs):
        if locs[idx]<locs[-1]:          # going from left to right
            b_left = max(0,locs[idx].numpy()-np.random.randint(20,80))
            b_right = b_bottom-b_top+b_left
        else:                          # going from right to left
            b_right = min(locs[idx].numpy()+np.random.randint(20,80),leng)
            b_left = b_right-b_bottom+b_top
        if b_left<0 or b_right>leng:
            break
        boxes.append((b_left,b_top,b_right,b_bottom))
        idxs.append(idx)
        idx+=b_in+b_out-b_ol
    return boxes,idxs