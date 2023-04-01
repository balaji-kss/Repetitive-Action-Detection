from act_dataset import TestDataset
import cv2
import matplotlib.pyplot as plt
import math
import torch
from model import ImageActNet
import act_dataset
from torchvision import transforms
import utils

def load_net(model_path, device):

    model = ImageActNet()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    return model

def load_data(image, joints, input_size):

    pre_img, pre_joints = act_dataset.preprocess_data(image, joints, input_size)

    pre_joints = torch.as_tensor(pre_joints, dtype=torch.float32)

    pre_img = transforms.ToTensor()(pre_img)
    pre_img = torch.unsqueeze(pre_img, 0)

    return pre_img, pre_joints

def predict(model, image, joints, input_size):

    pre_img, pre_joints = load_data(image, joints, input_size)
    pre_img = pre_img.to(device)
    out = model(pre_img)
    out = out[0].detach().cpu().numpy()
    out = round(float(out), 3)

    return out

def run(dataset, model_path, input_size, device):

    model = load_net(model_path, device)

    for fid in range(len(dataset)):

        frame = dataset[fid]["image"]
        joints = dataset[fid]["joints"]

        out = predict(model, frame, joints, input_size)
        disp_frame = utils.display_result(frame, out, fid, thresh)

        cv2.imshow('frame ', disp_frame)
        cv2.waitKey(-1)

if __name__ == "__main__":

    root_dir = "simple_data/lifting_1/"
    dataset = TestDataset(root_dir + "clip_3")
    model_path = '/home/balaji/Tumeke/models/' + root_dir + 'exp1/60.pth'
    input_size = 224
    device = "cuda"
    thresh = 0.6
    run(dataset, model_path, input_size, device)