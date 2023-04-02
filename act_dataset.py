# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import csv
import numpy as np
import cv2
import mmcv
import utils
import random
import os

def square_img(img):

    h, w, c = img.shape
    sz = max(h, w)

    sqr_img = np.zeros((sz, sz, c), dtype='uint8')
    sqr_img[:h, :w] = img
    
    return sqr_img

def preprocess_data(image, joints, input_res):
    
    norm_img = np.copy(image)

    norm_img = square_img(norm_img)
    norm_joints = normalize_joints(norm_img, joints)

    h, w, c = norm_img.shape

    for i in range(c//3):
        norm_img[:, :, i * 3 : (i + 1) * 3] = cv2.cvtColor(norm_img[:, :, i * 3 : (i + 1) * 3], cv2.COLOR_BGR2RGB)

    norm_img = cv2.resize(norm_img, (input_res, input_res)).astype(np.float32)
    norm_img /= 255.0
    
    return norm_img, norm_joints

def normalize_joints(img, joints):    

    norm_joints = np.copy(joints)
    h, w = img.shape[:2]
    norm_joints[:, 0] /= w
    norm_joints[:, 1] /= h

    return norm_joints

def unnormalize_data(norm_img, norm_joints):

    h, w = norm_img.shape[:2]
    img, joints = np.copy(norm_img), np.copy(norm_joints)
    img = np.clip(img * 255.0, 0, 255).astype('uint8')

    joints[:, 0] *= w
    joints[:, 1] *= h

    return img, joints

def vis_dataset(dataset):

    print('len dataset: ', len(dataset))
    for frameid in range(len(dataset)):
        
        img, joints, label = dataset[frameid]
        img, joints, label = img.numpy(), joints.numpy(), label.numpy()
        img = np.transpose(img, (1, 2, 0))

        h, w, c = img.shape
        for i in range(c//3):
            img[:, :, i * 3 : (i + 1) * 3] = cv2.cvtColor(img[:, :, i * 3 : (i + 1) * 3], cv2.COLOR_BGR2RGB)

        if c == 9:
            cimg = img[:, :, 3:6].copy()
            stack_img = np.hstack((img[:,:,6:], img[:,:,3:6]))
            stack_img = np.hstack((stack_img, img[:,:,:3]))
        else:
            cimg = img
    
        cimg, joints = unnormalize_data(cimg, joints)

        disp_img = utils.display_skeleton(cimg, joints)
        disp_img = utils.display_label(disp_img, int(label), frameid)

        if c == 9:
            cv2.imshow("stack image", stack_img)

        cv2.imshow("Test", disp_img)
        cv2.waitKey(-1)

class ActDataset(Dataset):
    """Class to load in pose data.
    Args:
        data_prefix (str): Path to a directory where pose data is held
    """

    def __init__(self, data_prefix, input_res, num_ts = 0, tstride = 0, mode="train"):
        
        self.mode = "train"
        self.train_ratio = 0.8
        self.input_res = input_res
        self.num_ts = num_ts
        self.tstride = tstride

        self.image_dir = f"{data_prefix}/images/"
        self.joints_2d_file = f"{data_prefix}/2d.csv"
        self.ground_truth_file = f"{data_prefix}/gt.csv"
        
        self.joints_2d = self.load_joint_data_2d()

        self.num_frames = len(self.joints_2d)
        self.neg_val = 0
        self.pos_val = 1
        self.fps = 16

        if self.mode == "train" or self.mode == "val":
            self.labels = self.load_gt()
            self.train_idxs = list(range(len(self.labels)))
            # self.sample_data()

    def sample_data(self):
        
        tot_samples = len(self.labels)
        pos_idxs = np.where(self.labels == self.pos_val)[0].tolist()
        neg_idxs = np.where(self.labels == self.neg_val)[0].tolist()
        
        print('tot_samples ', tot_samples)
        print('pos_samples ', len(pos_idxs))
        print('neg_samples ', len(neg_idxs))

        ## sampling
        random.seed(100)
        for i in range(5):
            random.shuffle(pos_idxs)
            random.shuffle(neg_idxs)

        min_len = min(len(pos_idxs), len(neg_idxs))
        pos_idxs = pos_idxs[:min_len]
        neg_idxs = neg_idxs[:min_len]
        sample_idxs = pos_idxs + neg_idxs
        
        for i in range(5):
            random.shuffle(sample_idxs)

        num_train = int(self.train_ratio * len(sample_idxs))
        self.train_idxs = sample_idxs[:num_train]
        self.val_idxs = sample_idxs[num_train:]
        
        print("*** after sampling ***")
        print('tot_samples ', len(sample_idxs))
        print('pos_samples ', len(pos_idxs))
        print('neg_samples ', len(neg_idxs))
        print('train_samples ', len(self.train_idxs))
        print('val_samples ', len(self.val_idxs))
        print('train ', self.train_idxs[:10])
        print('val ', self.val_idxs[:10])

    def get_bound_delta(self, durations, ratio = 0.05):
        
        mean_dur = np.mean(durations)
        print('mean duration ', mean_dur)

        delta = round(mean_dur * ratio * self.fps)
    
        return int(delta)

    def gen_labels(self, durations, end_time_stamps):
        
        delta = self.get_bound_delta(durations)
        print('delta ', delta)

        tot_time_stamps = [self.neg_val] * self.num_frames
        tot_time_stamps_np = np.array(tot_time_stamps)

        for fid in end_time_stamps:
            tot_time_stamps_np[fid:fid + delta] = self.pos_val

        return tot_time_stamps_np

    def load_gt(self):
        
        csvfile = open(self.ground_truth_file, newline='')
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        prev_ts = 0
        end_time_stamps, durations = [], []

        for row in reader:
            
            duration = float(row[0])
            prev_ts =  duration * self.fps + prev_ts
            durations.append(duration)
            int_prev_ts = int(round(prev_ts))
            end_time_stamps.append(int_prev_ts)

        csvfile.close()
        print('end_time_stamps ', end_time_stamps)
        labels = self.gen_labels(durations, end_time_stamps)

        return labels

    def load_joint_data_2d(self):

        data = []
        csvfile = open(self.joints_2d_file, newline='')
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            # Skip header
            if (i == 0):
                i += 1
                continue
            row_data = np.array([
                -1 if x == "nan" else float(x) for x in row[1:]
            ]).reshape([15, 2])
            data.append(row_data)
            i += 1
        csvfile.close()

        return np.array(data)

    def stack_temp_images(self, index):
        
        if self.num_ts == 1:
            img_path = os.path.join(self.image_dir, str(index) + '.jpg')
            img = cv2.imread(img_path)
            return img

        if self.num_ts == 3:

            tids = [index - self.tstride, index, index + self.tstride]
            stack_img = None
            
            for i in range(self.num_ts):
                tids[i] = min(max(0, tids[i]), self.num_frames - 1)
                img_path = os.path.join(self.image_dir, str(tids[i]) + '.jpg')
                img = cv2.imread(img_path)
                # print(tids[i], ' img shape ', img.shape)
                if stack_img is None:
                    stack_img = img
                else:
                    stack_img = np.concatenate((img, stack_img), axis=2)

            return stack_img

    def __len__(self):
        """Get the size of the dataset."""

        if self.mode == "train":
            return len(self.train_idxs)
        if self.mode == "val":
            return len(self.val_idxs)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        
        if self.mode == "train":
            index = self.train_idxs[idx]
        if self.mode == "val":
            index = self.val_idxs[idx]

        img = self.stack_temp_images(index)
        joints = self.joints_2d[index]
        label = self.labels[index]

        pre_img, pre_joints = preprocess_data(img, joints, self.input_res)

        input_img = transforms.ToTensor()(pre_img)        
        pre_joints = torch.as_tensor(pre_joints, dtype=torch.float32)
        label = torch.as_tensor([label], dtype=torch.float32)

        return input_img, pre_joints, label


class TestDataset(Dataset):
    """Class to load in pose data.
    Args:
        data_prefix (str): Path to a directory where pose data is held
    """

    def __init__(self, data_prefix):
        self.video_file = f"{data_prefix}/video.mp4"
        self.joints_2d_file = f"{data_prefix}/2d.csv"
        self.ground_truth_file = f"{data_prefix}/gt.csv"
        self.video = mmcv.VideoReader(self.video_file)
        self.joints_2d = self.load_joint_data_2d()

    def load_joint_data_2d(self):

        data = []
        csvfile = open(self.joints_2d_file, newline='')
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            # Skip header
            if (i == 0):
                i += 1
                continue
            row_data = np.array([
                -1 if x == "nan" else float(x) for x in row[1:]
            ]).reshape([15, 2])
            data.append(row_data)
            i += 1
        csvfile.close()

        return np.array(data)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.joints_2d)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        
        frame = self.video[idx]     
        return {
            "joints": self.joints_2d[idx],
            "image": frame
        }


if __name__ == "__main__":
	
    input_res = 360
    dataset = ActDataset("simple_data/lifting_1/clip_1", input_res, mode="train", num_ts = 3, tstride = 3)
    vis_dataset(dataset)
