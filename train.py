import torch
import numpy as np 
import cv2
import os
import torch.nn as nn
import time
import utils
from act_dataset import ActDataset, unnormalize_data
from model import ImageActNet, ImagePoseActNet
from torch.optim import lr_scheduler
# from torchsummary import summary

def data_loader(data_dir, input_size):

    train_set = ActDataset(data_dir, input_size, num_ts = num_ts, tstride = tstride, mode="train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = ActDataset(data_dir, input_size, num_ts = num_ts, tstride = tstride, mode="val")
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def vis_dataloader(dataloader):

    # Visualize data loader

    dataiter = iter(dataloader)
    imgs, joints, labels = dataiter.next()
    imgs, joints, labels = imgs.numpy(), joints.numpy(), labels.numpy()
    # (32, 9, 224, 224) (32, 30, 3)

    h, w = imgs.shape[-2:]
    imgs = imgs.reshape(-1, 3, 3, h, w)
    imgs = np.transpose(imgs, (0, 1, 3, 4, 2)).squeeze()
    joints = np.transpose(joints, (0, 2, 1))
    num_images = imgs.shape[0]
    # (32, 3, 224, 224, 3) (32, 3, 30)

    for bz in range(num_images):

        timgs, tjoints = imgs[bz], joints[bz]
        label = labels[bz]

        stack_img = None
        for i in range(timgs.shape[0]):
            img = timgs[i].copy()
            jts = tjoints[i].reshape(15, 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img, jts = unnormalize_data(img, jts)
            disp_img = utils.display_skeleton(img, jts)

            if stack_img is None:
                stack_img = disp_img 
            else:
                stack_img = np.hstack((disp_img, stack_img))

        stack_img = utils.display_label(stack_img, int(label), bz)
        cv2.imshow("stack_img", stack_img)
        cv2.waitKey(-1)

def train(train_loader, val_loader):

    # model = ImageActNet(inp_channels = inp_channels)
    model = ImagePoseActNet(inp_channels = inp_channels)
    model = model.to(device)
    # summary(model, (3, 224, 224))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    # regloss = torch.nn.SmoothL1Loss() 
    regloss = torch.nn.MSELoss() 

    print("[INFO] training the network...", flush=True)

    for epoch in range(num_epoch + 1):

        tloss_value = 0
        steps = 0
        startTime = time.time()
        model.train()

        for imgs, joints, labels in train_loader:
            
            optimizer.zero_grad()
            imgs = imgs.to(device)
            joints = joints.to(device)
            labels = labels.to(device)

            out = model(imgs, joints)
            
            loss = regloss(out, labels) 
            loss.backward()
            optimizer.step()

            tloss_value += loss.item()
            steps += 1

        endTime = time.time()	
        time_elapsed = (endTime - startTime) / 60 #mins
        avg_tloss_value = tloss_value / steps

        print("[INFO] EPOCH: {}/{}".format(epoch, num_epoch), flush=True)
        print("Time: {:.3f} min, Train loss: {:.6f}".format(
		time_elapsed, avg_tloss_value), flush=True)
        
        scheduler.step()
        
        print("[INFO] saving regression model...", flush=True)
        model_path = os.path.join(model_dir, str(epoch) + '.pth') 
        torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            print("[INFO] validate model...", flush=True)
            validate(val_loader, model)

def validate(val_loader, model):

    model.eval()
    tot_loss = 0 
    steps = 0
    regloss = torch.nn.SmoothL1Loss() 

    for imgs, joints, labels in val_loader:
            
        imgs = imgs.to(device)
        joints = joints.to(device)
        labels = labels.to(device)

        out = model(imgs, joints)

        loss = regloss(out, labels)
        tot_loss += loss.item()
        steps += 1

    avg_tloss_value = tot_loss / steps
    print("Val loss: {:.6f}".format(
		avg_tloss_value), flush=True)    

if __name__ == '__main__':
    
    batch_size = 32
    input_size = 224
    num_class = 1
    device = "cuda"
    lr = 1e-2
    num_epoch = 30
    num_ts = 3
    tstride = 3
    inp_channels = 3 * num_ts

    # root_dir = "hard_data/kontoor/"
    root_dir = "simple_data/lifting_2/"
    data_dir = root_dir + 'clip_1/'
    model_dir = './models/' + root_dir + 'exp3_4/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print('lr ', lr)
    print('data_dir: ', data_dir)
    print('model_dir: ', model_dir)

    train_loader, val_loader = data_loader(data_dir, input_size)
    
    vis_dataloader(train_loader)

    # train(train_loader, val_loader)