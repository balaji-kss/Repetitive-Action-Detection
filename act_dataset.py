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

def square_img(img):

    h, w = img.shape[:2]
    sz = max(h, w)

    sqr_img = np.zeros((sz, sz, 3), dtype='uint8')
    sqr_img[:h, :w] = img
    
    return sqr_img

def preprocess_data(image, joints, input_res):
    
    norm_img = np.copy(image)

    norm_img = square_img(norm_img)
    norm_joints = normalize_joints(norm_img, joints)

    norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    norm_img = cv2.resize(norm_img, (input_res, input_res))
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

    for frameid in range(len(dataset)):
        
        img, joints, label = dataset[frameid]
        img, joints, label = img.numpy(), joints.numpy(), label.numpy()
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        img, joints = unnormalize_data(img, joints)

        disp_img = utils.display_skeleton(img, joints)
        disp_img = utils.display_label(disp_img, int(label), frameid)

        cv2.imshow("Test", disp_img)
        cv2.waitKey(-1)

class ActDataset(Dataset):
    """Class to load in pose data.
    Args:
        data_prefix (str): Path to a directory where pose data is held
    """

    def __init__(self, data_prefix, input_res, train):
        
        self.train = train
        self.input_res = input_res
        self.video_file = f"{data_prefix}/video.mp4"
        self.joints_2d_file = f"{data_prefix}/2d.csv"
        self.ground_truth_file = f"{data_prefix}/gt.csv"
        
        self.joints_2d = self.load_joint_data_2d()
        self.video = mmcv.VideoReader(self.video_file)

        assert len(self.joints_2d) == len(self.video)
        self.num_frames = len(self.video)

        if self.train:
            self.labels = self.load_gt()

    def get_bound_delta(self, durations, ratio = 0.05):
        
        mean_dur = np.mean(durations)
        delta = round(mean_dur * ratio)
        
        print('durations ', durations)
        print('mean_dur ', mean_dur)
        print('delta ', delta)

        return int(delta)

    def gen_labels(self, durations, end_time_stamps):
        
        tot_time_stamps = [0] * self.num_frames
        tot_time_stamps_np = np.array(tot_time_stamps)
        tot_time_stamps_np[end_time_stamps] = 1

        return tot_time_stamps_np

    def load_gt(self):
        
        csvfile = open(self.ground_truth_file, newline='')
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        prev_ts = 0
        end_time_stamps, durations = [], []

        for row in reader:
            
            duration = float(row[0])
            prev_ts =  duration * self.video.fps + prev_ts
            durations.append(duration)
            int_prev_ts = int(round(prev_ts))
            end_time_stamps.append(int_prev_ts)

        csvfile.close()
        
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

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.joints_2d)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        
        img = self.video[idx]
        joints = self.joints_2d[idx]
        label = self.labels[idx]

        pre_img, pre_joints = preprocess_data(img, joints, self.input_res)

        input_img = transforms.ToTensor()(pre_img)        
        pre_joints = torch.as_tensor(pre_joints, dtype=torch.float32)
        label = torch.as_tensor([label], dtype=torch.float32)

        return input_img, pre_joints, label


if __name__ == "__main__":
	
    train = True
    input_res = 360
    dataset = ActDataset("simple_data/lifting_1/clip_1", input_res, train=train)
    vis_dataset(dataset)
