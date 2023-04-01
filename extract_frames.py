import cv2
import numpy as np
import os

video_path = '/home/balaji/Tumeke/simple_data/lifting_1/clip_1/video.mp4'
save_dir = '/home/balaji/Tumeke/simple_data/lifting_1/clip_1/images/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)    

    frames = []
    i = 0
    while True:

        ret, frame = cap.read()
        if ret is False: break

        img_path = os.path.join(save_dir, str(i) + '.jpg')
        cv2.imwrite(img_path, frame)

        i += 1

extract_frames(video_path)