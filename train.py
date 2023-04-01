import torch
import numpy as np 
import cv2
import os
import torch.nn as nn
import time
import utils
from act_dataset import ActDataset, unnormalize_data
from model import ImageActNet

def data_loader(data_dir, input_size):

    train_set = ActDataset(data_dir, input_size, train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    return train_loader

def vis_dataloader(dataloader):

    # Visualize data loader

    dataiter = iter(dataloader)
    imgs, joints, labels = dataiter.next()
    imgs, joints, labels = imgs.numpy(), joints.numpy(), labels.numpy()
    imgs = np.transpose(imgs, (0, 2, 3, 1)).squeeze()
    num_images = imgs.shape[0]

    for i in range(num_images):
        
        img, joint, label = imgs[i], joints[i], labels[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img, joint = unnormalize_data(img, joint)

        disp_img = utils.display_skeleton(img, joint)
        cv2.imshow("Test", disp_img)
        print('label ', int(label))

        cv2.waitKey(-1)

def train(train_loader):

    model = ImageActNet()
    model = model.to(config.DEVICE)
    summary(model, (3, 224, 224))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.INIT_LR)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    regloss = torch.nn.SmoothL1Loss() 

    print("[INFO] training the network...", flush=True)

    for epoch in range(num_epoch + 1):

        tloss_value = 0
        steps = 0
        startTime = time.time()

        for imgs, joints, labels in train_loader:
            
            optimizer.zero_grad()
            imgs = imgs.to(device)
            joints = joints.to(device)
            labels = labels.to(device)

            out = model(images)
            
            loss = regloss(out, labels) 
            loss.backward()
            optimizer.step()

            tloss_value += loss.item()
            steps += 1

        endTime = time.time()	
        time_elapsed = (endTime - startTime) / 60 #mins
        avg_tloss_value = tloss_value / steps

        print("[INFO] EPOCH: {}/{}".format(epoch, config.NUM_EPOCHS), flush=True)
        print("Time: {:.3f} min, Train loss: {:.6f}".format(
		time_elapsed, avg_tloss_value), flush=True)
        
        scheduler.step()
        
        print("[INFO] saving regression model...", flush=True)
        model_path = os.path.join(config.MODEL_DIR, str(epoch) + '.pth') 
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    
    batch_size = 64
    input_size = 224
    num_class = 1
    device = "cuda"
    lr = 1e-1
    num_epoch = 20

    data_dir = "simple_data/lifting_1/clip_1"
    model_dir = './models/'

    train_loader = data_loader(data_dir, input_size)
    
    # vis_dataloader(train_loader)

    train(train_loader)