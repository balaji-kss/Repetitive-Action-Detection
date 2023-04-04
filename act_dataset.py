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

    norm_img = square_img(norm_img) # (sz, sz, 9) joints: # ([15, 2]*3)
    
    norm_joints_lst = []
    for i in range(len(joints)):
        norm_joints = normalize_joints(norm_img, joints[i])
        norm_joints_lst.append(norm_joints)

    h, w, c = norm_img.shape

    for i in range(c//3):
        norm_img[:, :, i * 3 : (i + 1) * 3] = cv2.cvtColor(norm_img[:, :, i * 3 : (i + 1) * 3], cv2.COLOR_BGR2RGB)

    norm_img = cv2.resize(norm_img, (input_res, input_res)).astype(np.float32)
    norm_img /= 255.0
    
    return norm_img, norm_joints_lst

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
        img, joints, label = img.numpy(), joints.numpy(), label.numpy() # (9, 360, 360), (30, 3)
        
        h, w = img.shape[1:]
        img = img.reshape(3, 3, h, w)
        imgs = np.transpose(img, (0, 2, 3, 1)) # (3, sz, sz, 3)
        joints = np.transpose(joints, (1, 0)) # # (3, 30)
        num_imgs = imgs.shape[0]

        stack_img = None
        for i in range(num_imgs):
            img = imgs[i].copy()
            jts = joints[i].reshape(15, 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img, jts = unnormalize_data(img, jts)
            disp_img = utils.display_skeleton(img, jts)

            if stack_img is None:
                stack_img = disp_img 
            else:
                stack_img = np.hstack((disp_img, stack_img))

        stack_img = utils.display_label(stack_img, int(label), frameid)
        cv2.imshow("stack_img", stack_img)
        cv2.waitKey(-1)

def reshape_joints(joints_lst):

    if len(joints_lst) == 1:
        return joints_lst[0]

    if len(joints_lst) == 3:
        jtrl = [jt.reshape(30) for jt in joints_lst]
        return np.array(jtrl)

def get_joint_box(joints, pad = [0.7, 0.3]):

    joints_np = np.array(joints)
    joints_xs = joints_np[:, 0]
    joints_ys = joints_np[:, 1]

    joints_xs = joints_xs[joints_xs != -1]
    joints_ys = joints_ys[joints_ys != -1]

    sx, ex = np.min(joints_xs), np.max(joints_xs)
    sy, ey = np.min(joints_ys), np.max(joints_ys)

    w, h = ex - sx, ey - sy
    cx, cy = 0.5 * (sx + ex), 0.5 * (sy + ey)

    sx_, ex_ = cx - 0.5 * (1 + pad[0]) * w, cx + 0.5 * (1 + pad[0]) * w
    sy_, ey_ = cy - 0.5 * (1 + pad[1]) * h, cy + 0.5 * (1 + pad[1]) * h

    return [int(sx_), int(sy_), int(ex_), int(ey_)]

def get_patch_size(img_shape, sz_ratio = [0.2, 0.3], asp_ratio = [1, 3]):

    sz = np.max(img_shape)
    sz = random.uniform(sz_ratio[0], sz_ratio[1]) * sz
    asp = random.uniform(asp_ratio[0], asp_ratio[1])

    if random.uniform(0.0, 1.0 ) < 0.5:
        asp = 1./asp
    
    return sz * np.sqrt(asp), sz / np.sqrt(asp)

def bb_iou(boxA, boxB):
	
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou

def add_neg_patch(org_img, joints, num_try = 20, iou_thresh = 0.1, debug=0):

    img = org_img.copy()
    joints_bbox = get_joint_box(joints)
    sx, sy, ex, ey = joints_bbox
    h, w = img.shape[:2]

    if debug:
        cv2.rectangle(img, (sx, sy), (ex, ey), (255,0,0), 2)

    for i in range(num_try):
        px, py = random.randint(0, w//10) * 10, random.randint(0, 0.5 * h//10) * 10
        pw, ph = get_patch_size([h, w])
        psx, psy, pex, pey = px, py, px + pw, py + ph 
        # print('psx, psy, pex, pey ', psx, psy, pw, ph)
        iou = bb_iou([sx, sy, ex, ey], [psx, psy, pex, pey])
        # print('iou ', iou)
        if iou < iou_thresh:break

    bc, gc, rc = random.randint(0, 25) * 10, random.randint(0, 25) * 10, random.randint(0, 25) * 10
    psx, psy, pex, pey = int(psx), int(psy), int(pex), int(pey)
    cv2.rectangle(img, (psx, psy), (pex, pey), (bc,gc,rc), -1)

    if debug:
        cv2.imshow('img ', img)
        cv2.waitKey(-1)

    return img

class ActDataset(Dataset):
    """Class to load in pose data.
    Args:
        data_prefix (str): Path to a directory where pose data is held
    """

    def __init__(self, data_prefix, input_res, num_ts = 0, tstride = 0, mode="train"):
        
        self.mode = "train"
        self.train_ratio = 0.95
        self.input_res = input_res
        self.num_ts = num_ts
        self.tstride = tstride

        self.video_file = f"{data_prefix}/video.mp4"
        self.image_dir = f"{data_prefix}/images/"
        self.joints_2d_file = f"{data_prefix}/2d.csv"
        self.ground_truth_file = f"{data_prefix}/gt.csv"
        self.video = mmcv.VideoReader(self.video_file)

        self.joints_2d = self.load_joint_data_2d()

        self.num_frames = len(self.joints_2d)
        self.neg_val = 0
        self.pos_val = 3
        self.neg_ratio = 3
        self.fps = self.video.fps
        self.aug_ratio = 0.2
        print('self.fps ', self.fps)
        print('self.num_ts ', self.num_ts)
        print('self.tstride ', self.tstride)

        if self.mode == "train" or self.mode == "val":
            self.labels = self.load_gt()
            # self.train_idxs = list(range(len(self.labels)))
            self.sample_data()

    def sample_data(self):
        
        tot_samples = len(self.labels)
        pos_idxs = np.where(self.labels == self.pos_val)[0].tolist()
        neg_idxs = np.where(self.labels == self.neg_val)[0].tolist()
        
        print('tot_samples ', tot_samples)
        print('pos_samples ', len(pos_idxs))
        print('neg_samples ', len(neg_idxs))

        pos_len = len(pos_idxs)
        neg_len = int(self.neg_ratio * pos_len)
        neg_stride = max(1, len(neg_idxs) // neg_len)

        sample_pos_idxs = pos_idxs
        sample_neg_idxs = neg_idxs[::neg_stride]

        ## sampling
        random.seed(100)
        sample_idxs = sample_pos_idxs + sample_neg_idxs
        
        for i in range(5):
            random.shuffle(sample_idxs)

        num_train = int(self.train_ratio * len(sample_idxs))
        self.train_idxs = sample_idxs[:num_train]
        self.val_idxs = sample_idxs[num_train:]

        print("*** after sampling ***")
        print('neg_stride ', neg_stride)
        print('tot_samples ', len(sample_idxs))
        print('pos_samples ', len(sample_pos_idxs))
        print('neg_samples ', len(sample_neg_idxs))
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

    def stack_temp_data(self, index):
        
        if self.num_ts == 1:
            img_path = os.path.join(self.image_dir, str(index) + '.jpg')
            img = cv2.imread(img_path)
            joints = self.joints_2d[index]
            return img, [joints]

        if self.num_ts == 3:

            tids = [index - self.tstride, index, index + self.tstride]
            stack_img = None
            temp_joints = []

            for i in range(self.num_ts):
                tids[i] = min(max(0, tids[i]), self.num_frames - 1)
                img_path = os.path.join(self.image_dir, str(tids[i]) + '.jpg')
                img = cv2.imread(img_path)
                joints = self.joints_2d[tids[i]]
            
                # print(tids[i], ' img shape ', img.shape)
                if self.mode == "train" and random.uniform(0.0, 1.0) < self.aug_ratio:
                    img = add_neg_patch(img, joints)

                if stack_img is None:
                    stack_img = img
                else:
                    stack_img = np.concatenate((img, stack_img), axis=2)

                
                temp_joints.append(joints)
            
            temp_joints.reverse()

            return stack_img, temp_joints 

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

        img, joints_lst = self.stack_temp_data(index)
        label = self.labels[index]

        pre_img, pre_joints_lst = preprocess_data(img, joints_lst, self.input_res)
        pre_joints = reshape_joints(pre_joints_lst) # (3, 30)
        pre_joints = np.transpose(pre_joints, (1, 0)) # (30, 3)

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
    # inp_dir = "simple_data/lifting_2/clip_1"
    inp_dir = "hard_data/kontoor/clip_1"
    dataset = ActDataset(inp_dir, input_res, mode="train", num_ts = 3, tstride = 3)
    vis_dataset(dataset)
