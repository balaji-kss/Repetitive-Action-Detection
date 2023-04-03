import cv2
import numpy as np
import csv

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
        
def read_lst(lst_path):

    with open(lst_path, 'r') as ann_file:
        lines = ann_file.readlines()
        words = lines[0].rstrip().strip('\n').split(',')        
        
    return np.array([float(word) for word in words])

def get_durations(frame_confs, thresh):

    print('frame_confs ', frame_confs, frame_confs.shape)

def get_mov_avg(smooth_window, len_sw, cur_pred):

    smooth_window.append(cur_pred)
    if len(smooth_window) > len_sw:
        smooth_window.pop(0)
    
    avg = np.sum(smooth_window) / len(smooth_window)

    return round(avg, 3), smooth_window

def get_mov_avgs(len_sw, preds):

    smooth_win, pred_avgs = [], []
    for pred in preds:
        pred_avg, smooth_win = get_mov_avg(smooth_win, len_sw, pred)
        pred_avgs.append(pred_avg)

    return pred_avgs

def run(video_path, lst_path):

    cap = cv2.VideoCapture(video_path)    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps ', fps)
    len_sw = fps // 3

    frame_preds = read_lst(lst_path)
    get_durations(frame_preds, thresh)
    pred_avgs = get_mov_avgs(len_sw, frame_preds)

    i = 0
    
    while True:

        ret, frame = cap.read()
        if ret is False: break
    
        pred, pred_avg = frame_preds[i], pred_avgs[i]

        cv2.imshow('frame ', frame)
        print('pred ', pred, ' pred avg ', pred_avg)
        cv2.waitKey(-1)

        i += 1

if __name__ == "__main__":

    root_dir = './results/'
    video_path = root_dir + 'lifting_1_clip_2.mp4'
    lst_path = root_dir + 'lifting_1_clip_2.txt'
    thresh = 0.5

    run(video_path, lst_path)