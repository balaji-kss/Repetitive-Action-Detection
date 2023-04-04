import cv2
import numpy as np
import csv
import sys
import utils
np.set_printoptions(threshold=sys.maxsize)

def read_lst(lst_path):

    # Read prediction output from saved lst file

    with open(lst_path, 'r') as ann_file:
        lines = ann_file.readlines()
        words = lines[0].rstrip().strip('\n').split(',')        
        
    return np.array([float(word) for word in words])
    
def get_mov_avg(smooth_window, len_sw, cur_pred):

    # Get moving average to smoothen prediction

    smooth_window.append(cur_pred)
    if len(smooth_window) > len_sw:
        smooth_window.pop(0)
    
    avg = np.sum(smooth_window) / len(smooth_window)

    return round(avg, 3), smooth_window

def get_mov_avgs(len_sw, preds):

    # Get moving average for entire video

    smooth_win, pred_avgs = [], []
    for pred in preds:
        pred_avg, smooth_win = get_mov_avg(smooth_win, len_sw, pred)
        pred_avgs.append(pred_avg)

    return np.array(pred_avgs)

def get_intervals(pred_avgs):

    # Get detected task boundary intervals

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

    # Merge task boundary intervals

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

    # filter intervals based on interval length

    intervals_np = np.array(intervals)

    durs = intervals_np[:, 1] - intervals_np[:, 0]

    return intervals_np[durs >= valid_interval]

def get_durations(pred_avgs, conf_thresh, inter_dist, valid_interval):

    # Get durations of all the repetitive tasks

    pred_avgs[pred_avgs >= conf_thresh] = 1
    pred_avgs[pred_avgs < conf_thresh] = 0

    # get boundary intervals from prediction
    intervals = get_intervals(pred_avgs)

    # merge the intervals
    merged_intervals = merge_intervals(intervals, inter_dist)

    # filter intervals
    filtered_intervals = filter_intervals(merged_intervals, valid_interval)

    # remove outliers and calculate durations
    task_durations, filtered_intervals = get_task_dur(filtered_intervals)

    return task_durations, filtered_intervals

def label_frames(intervals, num_frames):

    # label frames

    labels = [0] * num_frames
    labels_np = np.array(labels)

    for interval in intervals:
        start, end = interval
        labels_np[start:end] = 1

    return labels_np

def remove_outliers(arr):

    # remove outliers 

    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr

    return np.where(arr < lower_bound)[0]

def get_task_dur(intervals):

    # get duration of all repetitive tasks

    intervals_np = np.array(intervals)

    if intervals_np[0, 0] > 5:
        intervals_np = np.insert(intervals_np, 0, [0, 0], axis=0)

    task_dur = intervals_np[1:, 0] - intervals_np[:-1, 0]

    # remove outliers
    idxs = remove_outliers(task_dur)
    fintervals_np = np.delete(intervals_np, idxs, axis=0)

    ftask_dur = fintervals_np[1:, 0] - fintervals_np[:-1, 0]

    return ftask_dur, fintervals_np

def write_csv(out_csv_file, labels):

    # write final durations to csv file
    
    with open(out_csv_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        for label in labels:
            writer.writerow([label])

def run(video_path, out_video_path, lst_path, out_csv_file, conf_thresh):

    # run post processing to estimate duration of repetitive tasks

    cap = cv2.VideoCapture(video_path)    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps, width, height = int(fps), int(width), int(height)

    sw_len = fps // 3   # smoothing window len
    valid_interval = fps // 2  # valid task boundary threshold
    inter_dist = fps # distance between two boundaries valid to merge
    
    # read raw predictions
    frame_preds = read_lst(lst_path)

    # get moving averages
    pred_avgs = get_mov_avgs(sw_len, frame_preds)
    
    assert len(frame_preds) == len(pred_avgs)
    num_frames = len(frame_preds)

    # get durations of all repetitive tasks
    task_durations, filtered_intervals = get_durations(pred_avgs, conf_thresh, inter_dist, valid_interval)
    task_durations_sec = task_durations / fps # convert to seconds

    # write to csv file
    write_csv(out_csv_file, task_durations_sec)

    # below code is for visual validation of task boundary detection
    # after post processing

    labels = label_frames(filtered_intervals, num_frames)
    i = -1
    if write_video:
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:

        ret, frame = cap.read()
        if ret is False: break
        i += 1

        utils.display_label(frame, i, pred_avgs[i], labels[i])

        if write_video:
            out.write(frame)
        else:
            cv2.imshow('frame ', frame)
            cv2.waitKey(-1)


if __name__ == "__main__":

    # paths
    root_dir = './results/'
    video_path = root_dir + 'kontoor_clip_2.mp4'
    lst_path = root_dir + 'kontoor_clip_2.txt'
    out_csv_file = root_dir + 'kontoor_clip_2.csv'
    conf_thresh = 0.3
    write_video = 1
    delta = 10
    out_video_path = video_path[:-4] + 'f.mp4'

    run(video_path, out_video_path, lst_path, out_csv_file, conf_thresh)