import cv2
import numpy as np
import csv

def read_lst(lst_path):

    with open(lst_path, 'r') as ann_file:
        lines = ann_file.readlines()
        words = lines[0].rstrip().strip('\n').split(',')        
        
    return np.array([float(word) for word in words])

def get_durations(frame_confs, thresh):

    print('frame_confs ', frame_confs, frame_confs.shape)


def read_video(video_path):

    cap = cv2.VideoCapture(video_path)    

    frames = []
    i = 0
    while True:

        ret, frame = cap.read()
        if ret is False: break
    
        i += 1

        # if i < 7300: continue
        # if i%3 != 0:continue

        # frame = display_label(frame, i)
        cv2.imshow('frame ', frame)
        cv2.waitKey(-1)

def display_label(image, frameid):

	cv2.putText(
		img = image,
		text = str(frameid),
		org = (20, 20),
		fontFace = cv2.FONT_HERSHEY_DUPLEX,
		fontScale = 0.5,
		color = (0, 0, 255),
		thickness = 1
	)

	return image

def load_gt(csv_file):
        
    csvfile = open(csv_file, newline='')
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    prev_ts = 0
    end_time_stamps, durations = [], []

    for row in reader:
        if len(row[0]) == 0:continue
        duration = float(row[0])
        prev_ts =  duration * 16 + prev_ts
        durations.append(duration)
        int_prev_ts = int(round(prev_ts))
        end_time_stamps.append(int_prev_ts)

    csvfile.close()

    return end_time_stamps
        

if __name__ == "__main__":

    root_dir = '/home/balajisundar/Documents/US/NEU/Full-time/Company/TuMeke/tumeke_takehome_final/tumeke_takehome_final/results_new/'
    video_path = root_dir + 'folding_clip_2.mp4'
    lst_path = root_dir + 'folding_clip_2.txt'
    thresh = 0.5

    frame_confs = read_lst(lst_path)
    get_durations(frame_confs, thresh)

    rvideo_path = '/home/balajisundar/Documents/US/NEU/Full-time/Company/TuMeke/tumeke_takehome_final/tumeke_takehome_final/results/kontoor_clip_2.mp4'

    # rvideo_path = './simple_data/lifting_3/clip_1/video.mp4'
    # csv_file = './simple_data/lifting_3/clip_1/gt.csv'

    # rvideo_path = './hard_data/kontoor/clip_1/video.mp4'
    # csv_file = './hard_data/kontoor/clip_1/gt.csv'
    # ets = load_gt(csv_file)
    # ets = [0] + ets
    # ets_np = np.array(ets)
    # print('diff ', ets_np[1:] - ets_np[:-1])
    # print('ets ', ets)
    # read_video(rvideo_path)