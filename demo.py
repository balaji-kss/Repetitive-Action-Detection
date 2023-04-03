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

    model = ImagePoseActNet(inp_channels = 3 * num_ts)
    # model = ImageActNet(inp_channels = 3 * num_ts)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    return model

def load_data(image, joints, input_size):

    pre_img, pre_joints = act_dataset.preprocess_data(image, joints, input_size)
    pre_joints = act_dataset.reshape_joints(pre_joints) #(224, 224, 9) (3, 30)
    pre_joints = np.transpose(pre_joints, (1, 0)) #(30, 3)
    pre_joints = torch.as_tensor(pre_joints, dtype=torch.float32)

    pre_img = transforms.ToTensor()(pre_img)
    pre_img = torch.unsqueeze(pre_img, 0)
    pre_joints = torch.unsqueeze(pre_joints, 0)

    return pre_img, pre_joints

def predict(model, image, joints, input_size):

    pre_img, pre_joints = load_data(image, joints, input_size)

    # vis_input(pre_img, pre_joints)

    pre_img = pre_img.to(device)
    pre_joints = pre_joints.to(device)
    out = model(pre_img, pre_joints)
    # out = model(pre_img)
    out = out[0].detach().cpu().numpy() / pos_val
    # out = min(max(0, out), 1)
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
            # print('num_frames, tid', num_frames, tids[i])
            img = dataset[tids[i]]["image"]
            joints = dataset[tids[i]]["joints"]

            if img is None:
                return None, None, None

            if stack_img is None:
                stack_img = img
            else:
                stack_img = np.concatenate((img, stack_img), axis=2)
            temp_joints.append(joints)
        
        temp_joints.reverse()

        return stack_img, stack_img[:, :, 3:6].copy(), temp_joints

def vis_input(imgs, joints):

    # (1, 9, 224, 224) (1, 30, 3)
    imgs, joints = imgs.numpy()[0], joints.numpy()[0]
    # (9, 224, 224) (30, 3)

    h, w = imgs.shape[1:]
    imgs = imgs.reshape(3, 3, h, w)
    imgs = np.transpose(imgs, (0, 2, 3, 1)) # (3, 224, 224, 3)
    joints = np.transpose(joints, (1, 0)) # # (3, 30)
    num_imgs = imgs.shape[0]

    stack_img = None
    for i in range(num_imgs):
        img = imgs[i].copy()
        jts = joints[i].reshape(15, 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img, jts = act_dataset.unnormalize_data(img, jts)
        disp_img = utils.display_skeleton(img, jts)

        if stack_img is None:
            stack_img = disp_img 
        else:
            stack_img = np.hstack((disp_img, stack_img))

    cv2.imshow("stack_img", stack_img)
    cv2.waitKey(-1)

def get_mov_avg(smooth_window, len_sw, cur_pred):

    smooth_window.append(cur_pred)
    if len(smooth_window) > len_sw:
        smooth_window.pop(0)
    
    avg = np.sum(smooth_window) / len(smooth_window)

    return round(avg, 3), smooth_window

def run(dataset, model_path, input_size, device):

    model = load_net(model_path, device)

    if write_video:
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 16, (270, 480))

    frame_confs = []
    smooth_win = []
    for fid in range(len(dataset)):

        if fid < 3000: continue

        input_, disp_frame, joints = stack_data(dataset, fid)

        if input_ is None:continue

        pred = predict(model, input_, joints, input_size)
        
        prev_avg, smooth_win = get_mov_avg(smooth_win, len_sw, pred)

        # print('smooth_win ', smooth_win)
        frame_confs.append(pred)

        disp_frame = utils.display_result(disp_frame, pred, prev_avg, fid, thresh)

        if write_video:
            out.write(disp_frame)
        else:
            cv2.imshow('frame ', disp_frame)
            cv2.waitKey(-1)

    if write_video:
        write_lst(save_lst_path, frame_confs)

def write_lst(lst_path, frame_confs):
    
    with open(lst_path, 'w+') as f:    
        f.write(','.join(str(j) for j in frame_confs))
        f.write('\n')
        f.flush()

if __name__ == "__main__":

    clip = "clip_3"
    root_dir = "simple_data/lifting_2/"
    # root_dir = "hard_data/kontoor/"
    inp_video_dir = root_dir + clip + "/"
    exp = 'exp3_4'
    model_path = './models/' + root_dir + '/' + exp + '/30.pth'
    out_video_dir = inp_video_dir + exp + '/'
    act_name = root_dir.rsplit('/')[1] + '_'
    out_video_path = out_video_dir + act_name + clip + '.mp4'
    save_lst_path = out_video_dir + act_name + clip + '.txt'

    if not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir)
        
    print('inp_video_dir  ', inp_video_dir)
    print('out_video_path  ', out_video_path)
    print('save_lst_path  ', save_lst_path)
    print('model_path ', model_path, flush=True)

    input_size = 224
    device = "cuda"
    thresh = 0.5
    num_ts = 3
    tstride = 3
    write_video = 0
    pos_val = 3
    
    dataset = TestDataset(inp_video_dir)

    len_sw = dataset.video.fps // 3 # 0.5 sec smoothing

    run(dataset, model_path, input_size, device)