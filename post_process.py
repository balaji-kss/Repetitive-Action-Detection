from re import L
import cv2
import numpy as np
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)

def display_label(image, frameid, label):

    if label == 1:
        color = (0, 0, 255)
        text = str(frameid) + " : task ended"
    else:
        color = (255, 0, 0)
        text = str(frameid) + " : task not ended"

    cv2.putText(
        img = image,
        text = text,
        org = (20, 40),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.5,
        color = color,
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

    return np.array(pred_avgs)

def get_intervals(pred_avgs):

    intervals = []
    start = None
    for i in range(len(pred_avgs)):
        if pred_avgs[i] == 1:
            if start is None:
                start = i
        elif start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(pred_avgs)-1))

    return intervals

def merge_intervals(intervals, inter_dist):

    num_intervals = len(intervals)
    out = [intervals[0]]
    i = 1

    while i <  num_intervals:
        os, oe = out[-1]
        s, e = intervals[i]
        if s - oe <= inter_dist:
            os = min(os, s)
            oe = max(oe, e)
            out[-1] = [os, oe]
        else:
            out.append([s, e])
        i+=1
        
    return out

def filter_intervals(intervals, valid_interval):

    intervals_np = np.array(intervals)

    durs = intervals_np[:, 1] - intervals_np[:, 0]

    return intervals_np[durs >= valid_interval]

def get_durations(frame_preds, pred_avgs, conf_thresh, inter_dist, valid_interval):

    pred_avgs[pred_avgs >= conf_thresh] = 1
    pred_avgs[pred_avgs < conf_thresh] = 0

    intervals = get_intervals(pred_avgs)
    print('intervals ', intervals, len(intervals))

    merged_intervals = merge_intervals(intervals, inter_dist)
    print('merged_intervals ', merged_intervals, len(merged_intervals))

    filtered_intervals = filter_intervals(merged_intervals, valid_interval)
    print('filtered_intervals ', filtered_intervals, len(filtered_intervals))

    task_durations, filtered_intervals = get_task_dur(filtered_intervals)

    return task_durations, filtered_intervals

def label_frames(intervals, num_frames):

    labels = [0] * num_frames
    labels_np = np.array(labels)

    for interval in intervals:
        start, end = interval
        labels_np[start:end] = 1

    return labels_np

def remove_outliers(arr):
    
    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.25 * iqr
    print('lower_bound ', lower_bound)

    return np.where(arr < lower_bound)[0]

def get_task_dur(intervals):

    intervals_np = np.array(intervals)

    if intervals_np[0, 0] > 5:
        intervals_np = np.insert(intervals_np, 0, [0, 0], axis=0)

    boundary_dur = intervals_np[:, 1] - intervals_np[:, 0]
    task_dur = intervals_np[1:, 0] - intervals_np[:-1, 0]

    print('boundary_dur ', boundary_dur)
    print('task_dur ', task_dur, task_dur.shape)
    print('before task_dur min max ', np.min(task_dur), np.max(task_dur))

    # remove outliers
    idxs = remove_outliers(task_dur)
    
    fintervals_np = np.delete(intervals_np, idxs, axis=0)
    print('fintervals_np ', fintervals_np)

    ftask_dur = fintervals_np[1:, 0] - fintervals_np[:-1, 0]

    validate(ftask_dur, fintervals_np)

    return ftask_dur, fintervals_np

def validate(task_dur, intervals_np):

    ftask_dur, fintervals_np = task_dur.copy(), intervals_np.copy()
    ftask_dur = np.insert(ftask_dur, 0, 0, axis=0)

    val = np.concatenate((fintervals_np, ftask_dur.reshape(-1, 1)), axis=1)

    print('min max dur ', np.min(ftask_dur[1:]), np.max(ftask_dur), np.sort(ftask_dur))
    print('val ', val, val.shape)
    
def write_csv(out_csv_file, labels):

    with open(out_csv_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        for label in labels:
            writer.writerow([label])

def run(video_path, out_video_path, lst_path, out_csv_file, conf_thresh):

    cap = cv2.VideoCapture(video_path)    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps, width, height = int(fps), int(width), int(height)
    print('fps, width, height ', fps, width, height)

    sw_len = fps // 10
    valid_interval = fps // 10
    inter_dist = fps
      
    frame_preds = read_lst(lst_path)
    pred_avgs = get_mov_avgs(sw_len, frame_preds)
    
    assert len(frame_preds) == len(pred_avgs)
    num_frames = len(frame_preds)
    print('num_frames ', num_frames)

    task_durations, filtered_intervals = get_durations(frame_preds, pred_avgs, conf_thresh, inter_dist, valid_interval)
    task_durations_sec = task_durations / fps

    write_csv(out_csv_file, task_durations_sec)

    labels = label_frames(filtered_intervals, num_frames)

    i = -1

    if write_video:
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:

        ret, frame = cap.read()
        if ret is False: break
        i += 1
        
        # if i % 3 != 0: continue

        # if i < 8000:continue
        if i >= num_frames:break
        
        pred, pred_avg = frame_preds[i], pred_avgs[i]

        display_label(frame, i, labels[i])

        if write_video:
            out.write(frame)
        else:
            cv2.imshow('frame ', frame)
            cv2.waitKey(-1)


if __name__ == "__main__":

    inp_dir = '/home/balajisundar/Documents/US/NEU/Full-time/Company/TuMeke/tumeke_takehome_final/'
    root_dir = './results/'

    inp_video_path = inp_dir + 'hard_data/kontoor/clip_2/video.mp4'
    # inp_video_path = inp_dir + 'simple_data/lifting_3/clip_2/video.mp4'
    video_path = root_dir + 'kontoor_clip_2.mp4'
    lst_path = root_dir + 'kontoor_clip_2.txt'
    out_csv_file = root_dir + 'kontoor_clip_2.csv'
    conf_thresh = 0.35
    write_video = 0
    out_video_path = video_path[:-4] + 'f.mp4'
    print('out_video_path ', out_video_path)

    run(video_path, out_video_path, lst_path, out_csv_file, conf_thresh)