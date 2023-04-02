from act_dataset import TestDataset
import cv2
import matplotlib.pyplot as plt
import math
import torch
from model import ImageActNet, ImagePoseActNet
import act_dataset
from torchvision import transforms
import utils
import numpy as np
import os

def load_net(model_path, device):

    model = ImagePoseActNet(inp_channels=3*num_ts)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    return model

def load_data(image, joints, input_size):

    pre_img, pre_joints = act_dataset.preprocess_data(image, joints, input_size)
    pre_joints = act_dataset.reshape_joints(pre_joints)
    print('pre_img, pre_joints ', pre_img.shape, pre_joints.shape)
    pre_joints = np.transpose(pre_joints, (1, 0))
    pre_joints = torch.as_tensor(pre_joints, dtype=torch.float32)

    pre_img = transforms.ToTensor()(pre_img)
    pre_img = torch.unsqueeze(pre_img, 0)
    pre_joints = torch.unsqueeze(pre_joints, 0)

    return pre_img, pre_joints

def predict(model, image, joints, input_size):

    pre_img, pre_joints = load_data(image, joints, input_size)
    pre_img = pre_img.to(device)
    pre_joints = pre_joints.to(device)
    out = model(pre_img, pre_joints)
    out = out[0].detach().cpu().numpy()
    out = min(max(0, out), 1)
    out = round(float(out), 3)

    return out

def stack_data(dataset, fid):

    num_frames = len(dataset)
    
    if num_ts == 1:
        frame = dataset[fid]["image"]
        joints = dataset[fid]["joints"]
        return frame, frame, joints
    
    elif num_ts == 3:
        tids = [fid - tstride, fid, fid + tstride]
        stack_img = None
        temp_joints = []

        for i in range(num_ts):
            tids[i] = min(max(0, tids[i]), num_frames - 1)
            img = dataset[tids[i]]["image"]
            joints = dataset[fid]["joints"]
            if stack_img is None:
                stack_img = img
            else:
                stack_img = np.concatenate((img, stack_img), axis=2)
            temp_joints.append(joints)
        
        temp_joints.reverse()

        return stack_img, stack_img[:, :, 3:6].copy(), temp_joints


def run(dataset, model_path, input_size, device):

    model = load_net(model_path, device)

    if write_video:
        print('save video path: ', out_video_path)
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 16, (720, 480))

    for fid in range(len(dataset)):

        input_, disp_frame, joints = stack_data(dataset, fid)

        pred = predict(model, input_, joints, input_size)
        disp_frame = utils.display_result(disp_frame, pred, fid, thresh)

        if write_video:
            out.write(disp_frame)
        else:
            cv2.imshow('frame ', disp_frame)
            cv2.waitKey(-1)

if __name__ == "__main__":

    # root_dir = "simple_data/lifting_1/"
    root_dir = "hard_data/folding/"
    inp_video_dir = root_dir + "clip_2/"
    exp = 'exp3'
    model_path = './models/' + root_dir + '/' + exp + '/60.pth'
    out_video_dir = inp_video_dir + exp + '/'
    out_video_path = out_video_dir + 'res.mp4'

    if not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir)
        
    print('inp_video_dir  ', inp_video_dir)
    print('out_video_path  ', out_video_path)
    print('model_path ', model_path)

    input_size = 224
    device = "cuda"
    thresh = 0.6
    num_ts = 3
    tstride = 3
    write_video = 0

    dataset = TestDataset(inp_video_dir)
    run(dataset, model_path, input_size, device)