from pose_dataset import PoseDataset
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import math

def display_skeleton(data, image):
	for i in range(data.shape[0]):
		if (math.isnan(data[i][0]) or math.isnan(data[i][1])):
			continue
		image = cv2.circle(image, (int(data[i][0]), int(data[i][1])), 3, (255, 255, 255), 2)

	return image

if __name__ == "__main__":
	dataset = PoseDataset("hard_data/folding/clip_1")

	loader = DataLoader(dataset)
	for sample in loader:
		img = sample["image"].numpy()
		cv2.imshow("Test", display_skeleton(sample["joints"].numpy()[0], img[0]))

		cv2.waitKey(0)
