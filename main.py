from pose_dataset import PoseDataset
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import math

def display_skeleton(image, data):

	for i in range(data.shape[0]):
		if (math.isnan(data[i][0]) or math.isnan(data[i][1])):
			continue
		image = cv2.circle(image, (int(data[i][0]), int(data[i][1])), 3, (0, 0, 255), 2)

	return image

def display_label(image, label, frameid):

	cv2.putText(
		img = image,
		text = str(frameid) + " : " + str(label),
		org = (40, 40),
		fontFace = cv2.FONT_HERSHEY_DUPLEX,
		fontScale = 1.0,
		color = (0, 0, 255),
		thickness = 1
	)

	return image

def vis_loader(dataset):

	for frameid in range(len(dataset)):
		
		sample = dataset[frameid]
		img = sample["image"]

		if train:
			label = sample["label"]
		else:
			label = 0

		joints = sample["joints"]

		disp_img = display_skeleton(img, joints)
		display_label(disp_img, label, frameid)

		cv2.imshow("Test", disp_img)

		cv2.waitKey(-1)

if __name__ == "__main__":
	
	train = True
	dataset = PoseDataset("simple_data/lifting_1/clip_1", train=train)

	loader = DataLoader(dataset)

	vis_loader(dataset)
