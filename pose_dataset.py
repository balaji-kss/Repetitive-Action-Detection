# Copyright (c) OpenMMLab. All rights reserved.
import copy
from torch.utils.data import Dataset
import csv
import numpy as np
import cv2
import mmcv

class PoseDataset(Dataset):
    """Class to load in pose data.
    Args:
        data_prefix (str): Path to a directory where pose data is held
    """

    def __init__(self, data_prefix):
        self.video_file = f"{data_prefix}/video.mp4"
        self.joints_2d_file = f"{data_prefix}/2d.csv"
        self.ground_truth_file = f"{data_prefix}/gt.csv"

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
                np.nan if x == "nan" else float(x) for x in row[1:]
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
        video = mmcv.VideoReader(self.video_file)
        frame = video[idx]     
        return {
            "joints": self.joints_2d[idx],
            "image": frame
        }

