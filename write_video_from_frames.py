import cv2
import os
import sys
import natsort

IMAGE_DIR = 'input_frames_path'
VIDEO_OUTPUT = 'video.avi'
WIDTH = 640
HEIGHT = 360
FPS = 240


def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.2f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():
	IMAGE_PATH = [os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if x.endswith('.png')]
	IMAGE_PATH = natsort.natsorted(IMAGE_PATH)
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

	for img_path in IMAGE_PATH:
	    # Read images from folder (BGR order)
	    try:
	        img = cv2.imread(img_path) 
	    except IOError:
	        print('Cannot retrieve image. Please check path: ' + path)
	        return

	    current_frame_number = current_frame_number + 1
	    
	    # Write the image in the video
	    video_writer.write(img)

	    # Display progress
	    progress_bar(current_frame_number / num_frames * 100, bar_length=30)

	# Close the video writer
	video_writer.release()


if __name__ == "__main__":
    main()
