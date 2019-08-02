"""Data utility functions."""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image, ImageChops, ImageStat
from torchvision import transforms
import cv2


class FramesData(data.Dataset):
    """
    Implements pytorch Dataset.
    """

    def __init__(self, block_in, block_out, folder_path):
        self.root_dir = folder_path  # check out how to do this elegantly
        self.block_in = block_in
        self.block_out = block_out

        self.all_trials = sorted([os.path.join(self.root_dir, name) for name in
                                  os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, name))])

    def __getitem__(self, key):
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
        images_future = all_images[self.block_in:self.block_in + self.block_out]

        to_tensor = transforms.ToTensor()

        imgs = [Image.open(os.path.join(folder_to_read, x)) for x in images_past]
        imgs = torch.stack([to_tensor(img) * 2 - 1 for img in imgs], dim=1)

        targets = [Image.open(os.path.join(folder_to_read, x)) for x in images_future]
        targets = torch.stack([to_tensor(img) * 2 - 1 for img in targets], dim=1)

        return imgs, targets


def transform_UCF_dataset(block_in, block_out, shift, skip, folder_path, folder_name_out):
    """
    Splits UCF videos into model input sequences.
    :param block_in: the number of input frames
    :param block_out: the number of output frames
    :param shift: the shift between sequences
    :param skip: number of skip frames
    :param folder_path: input path
    :param folder_name_out: output path
    :return: None
    """
    save_dir = os.path.normpath(folder_path + os.sep + os.pardir + os.sep + folder_name_out)
    os.mkdir(save_dir)
    with open(os.path.join(folder_path, 'classes_red.txt'), 'r') as file:
        video_folders = [os.path.join(folder_path, line[:-1]) for line in file]
        all_videos = sorted([os.path.join(x, y) for x in video_folders for y in os.listdir(x)])

        frame_cnt = []
        for video_path in all_videos:
            vidcap = cv2.VideoCapture(video_path)
            frame_cnt.append(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))

        n_blocks = []
        seglen = block_in + block_out
        for cnt in frame_cnt:
            n_blocks.append((cnt - (seglen - 1) * skip - 1) // shift)

        to_image = transforms.ToPILImage()

        print('Dataset length will be: {}'.format(sum(n_blocks)))

        for index in range(sum(n_blocks)):
            all_blocks = np.cumsum(n_blocks)
            video_idx = np.searchsorted(all_blocks, index)
            if video_idx == 0:
                block_idx = index
            else:
                block_idx = index - all_blocks[video_idx - 1] - 1
            frame_idx = [block_idx * shift]
            for k in range(seglen - 1):
                frame_idx.append(frame_idx[-1] + skip)

            vidcap = cv2.VideoCapture(all_videos[video_idx])

            imgs = []
            total_success = True
            for k in range(len(frame_idx)):
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx[k])
                success, image = vidcap.read()
                total_success &= success
                # switch R and B (cv2 - BGR, normal - RGB)
                imgs.append(image)

            if total_success:
                new_set = os.path.join(save_dir, '%06d' % index)
                os.mkdir(new_set)
                for k in range(len(imgs)):
                    im = cv2.cvtColor(imgs[k], cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im[:, 40:280, :], (64, 64))
                    to_image(im).save(os.path.join(new_set, '%02d.png' % k))


def transform_KITTI_dataset(block_in, block_out, overlap, path_old):
    '''
    Splits KITTI videos into model input sequences.
    :param block_in: the number of input frames
    :param block_out: the number of output frames
    :param overlap: the overlap between sequences from the same video
    :param path_old: the input path
    :return: None
    '''
    path_up = os.path.normpath(path_old + os.sep + os.pardir)
    path_new = os.path.join(path_up, "in_%d_out_%d_ol_%d" % (block_in, block_out, overlap))
    os.mkdir(path_new)

    to_tensor = transforms.ToTensor()

    all_trials = sorted([os.path.join(path_old, name) for name in
                         os.listdir(path_old) if os.path.isdir(os.path.join(path_old, name))])
    all_trials = [trial + '/image_02/data' for trial in all_trials]
    sets = 0
    for folder_to_read in all_trials:
        all_images = sorted(os.listdir(folder_to_read))
        # read in all the images
        imgs = [Image.open(os.path.join(folder_to_read, x)) for x in all_images]
        # calculate diffs between images for movement detection
        img_diffs = [ImageChops.difference(a, b) for a, b in zip(imgs, imgs[1:])]
        # transform diffs to tensors for easier manipulation
        img_diff_tens = [to_tensor(x).mean(dim=0) for x in img_diffs]
        # sum the pixels over longer edge (shows where movement is)
        hists = [x.sum(dim=0) for x in img_diff_tens]
        locs = [x.argmax() for x in hists]

        hists_y = [x.sum(dim=1) for x in img_diff_tens]
        locs_y = [x.argmax() - 100 for x in hists_y]
        # generate crop boxes
        boxes, idxs = generate_boxes(locs, img_diff_tens[0].shape[1], img_diff_tens[0].shape[0], np.mean(locs_y),
                                     block_in, block_out,
                                     overlap)
        for start, box in zip(idxs, boxes):
            new_set = os.path.join(path_new, '%05d' % sets)
            os.mkdir(new_set)
            sets += 1
            for idx in range(start, start + (block_in + block_out)):
                new_img = imgs[idx].crop(box).resize((64, 64))
                new_img.save(os.path.join(new_set, '%02d.png' % idx))


def generate_boxes(locs, leng, heig, max_y, b_in, b_out, b_ol):
    boxes = []
    idxs = []
    b_top = max_y
    b_bottom = b_top + 128
    if b_bottom > heig:
        b_bottom = heig
        b_top = b_bottom - 128

    idx = 0
    while idx + (b_in + b_out - b_ol) < len(locs):
        if locs[idx] < locs[-1]:  # going from left to right
            b_left = max(0, locs[idx].numpy() - np.random.randint(5, 20))
            b_right = b_bottom - b_top + b_left
            if b_right > leng:
                b_right = leng
                b_left = b_right - b_bottom + b_top
        else:  # going from right to left
            b_right = min(locs[idx].numpy() + np.random.randint(5, 20), leng)
            b_left = b_right - b_bottom + b_top
            if b_left < 0:
                b_left = 0
                b_right = b_left + b_bottom - b_top
        boxes.append((b_left, b_top, b_right, b_bottom))
        idxs.append(idx)
        idx += (b_in + b_out - b_ol)
    return boxes, idxs
