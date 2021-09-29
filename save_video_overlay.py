import cv2
import os
import sys
import natsort

NUM_VIDEO = 'vid7'
IMAGE_DIR = 'video/' + NUM_VIDEO 
PRED_DIR = 'results/video/' + NUM_VIDEO 
VIDEO_OUTPUT = 'results/video/' + NUM_VIDEO + '_overlay.avi'
WIDTH = 640 
HEIGHT = 360
FPS = 60


def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():
	IMAGE_PATH = [os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if x.endswith('.png') or x.endswith('.jpg')]
	IMAGE_PATH = natsort.natsorted(IMAGE_PATH)

	PRED_PATH = [os.path.join(PRED_DIR, x) for x in os.listdir(PRED_DIR) if x.endswith('.png')]
	PRED_PATH = natsort.natsorted(PRED_PATH)

	video_writer = cv2.VideoWriter(str(VIDEO_OUTPUT),
	                               cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
	                               FPS,
	                               (WIDTH, HEIGHT))

	if not video_writer.isOpened():
	    sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
	                     "permissions.\n")
	    exit()

	num_frames = len(IMAGE_PATH)
	current_frame_number=0

	for current_frame_number in range(num_frames):
	    # Read data from folder (BGR order)
	    try:
	        img = cv2.imread(IMAGE_PATH[current_frame_number]) 
	        pred = cv2.imread(PRED_PATH[current_frame_number])
	    except IOError:
	        print('Cannot retrieve image. Please check path: ' + IMAGE_PATH)
	        return

	    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
	    alpha = 0.5
	    overlay = cv2.addWeighted(img, alpha, pred, 1-alpha, 0.0)
	    
	    # Write the overlay image in the video
	    # cv2.imwrite assumes the color order of the input image is BGR and saves it in RGB order. 
	    # Since overlay is already in RGB, change to BGR to save it correctly.
	    video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

	    # Display progress
	    progress_bar((current_frame_number+1) / num_frames * 100, bar_length=30)

	# Close the video writer
	video_writer.release()


if __name__ == "__main__":
    main()
