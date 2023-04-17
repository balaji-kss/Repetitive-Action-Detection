import cv2
import os

def display_skeleton(image, data):

    # display skeleton joints in the image

	for i in range(data.shape[0]):
		if data[i][0] == -1 or data[i][1] == -1:
			continue
		image = cv2.circle(image, (int(data[i][0]), int(data[i][1])), 3, (0, 0, 255), 2)

	return image

def display_label(image, label, frameid):

    # display label in the image

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

def extract_frames(video_path, save_dir):

    # Extract all frames in a video

    cap = cv2.VideoCapture(video_path)    

    frames = []
    i = 0
    while True:

        ret, frame = cap.read()
        if ret is False: break

        img_path = os.path.join(save_dir, str(i) + '.jpg')
        cv2.imwrite(img_path, frame)

        i += 1

def display_result(image, conf, sm_conf, frameid, thresh):

    # display results

    if sm_conf >= thresh:
        status = "task ended"
        color = (0, 0, 255)
    else:
        status = "task not ended"
        color = (255, 0, 0)

    text_ = str(frameid) + " : " + str(conf) + " : "  + str(sm_conf) + " : " + status

    cv2.putText(
        img = image,
        text = text_,
        org = (20, 20),
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.5,
        color = color,
        thickness = 1
    )

    return image

if __name__ == "__main__":

    video_path = 'video.mp4'
    save_dir = 'images/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    extract_frames(video_path, save_dir)