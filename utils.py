from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import math

def display_skeleton(image, data):

	for i in range(data.shape[0]):
		if data[i][0] == -1 or data[i][1] == -1:
			continue
		image = cv2.circle(image, (int(data[i][0]), int(data[i][1])), 3, (0, 0, 255), 2)

	return image

def display_label(image, label, frameid):

	cv2.putText(
		img = image,
		text = str(frameid) + " : " + str(label),
		org = (20, 20),
		fontFace = cv2.FONT_HERSHEY_DUPLEX,
		fontScale = 0.5,
		color = (0, 0, 255),
		thickness = 1
	)

	return image