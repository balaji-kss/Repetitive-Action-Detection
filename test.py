import cv2
import numpy as np
import csv

def visualize(video_path, csv_path):

    cap = cv2.VideoCapture(video_path)    
    fps = cap.get(cv2.CAP_PROP_FPS)

    end_time_stamps = load_gt(csv_path, fps)
    print('end_time_stamps ', end_time_stamps)

    i = -1
    while True:

        ret, frame = cap.read()
        if ret is False: break
        i += 1
        if i % 3!=0:continue

        # frame = display_label(frame, i, end_time_stamps)

        cv2.imshow('frame ', frame)
        cv2.waitKey(-1)

def display_label(image, frameid, end_time_stamps):

    if frameid in end_time_stamps:
        color = (0, 0, 255)
        text = str(frameid) + " : BOUNDARY"
    else:
        color = (255, 0, 0)
        text = str(frameid) + " : NO BOUNDARY"

    cv2.putText(
        img = image,
        text = text,
        org = (20, 60),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.8,
        color = color,
        thickness = 1
    )

    return image

def load_gt(csv_path, fps):
        
    csvfile = open(csv_path, newline='')
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    prev_ts = 0
    end_time_stamps, durations = [], []

    for row in reader:
        duration = float(row[0])
        prev_ts =  duration * fps + prev_ts
        durations.append(duration)
        int_prev_ts = int(round(prev_ts))
        end_time_stamps.append(int_prev_ts)

    csvfile.close()

    return end_time_stamps

if __name__ == "__main__":

    video_path = 'video.mp4'
    csv_path = 'video.csv'

    visualize(video_path, csv_path)