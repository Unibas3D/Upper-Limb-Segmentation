import cv2
import os

VIDEO_NUM = 'vid2'
VIDEO_FILE = 'video/' + VIDEO_NUM + '.mp4'
FRAME_FOLDER = 'video/' + VIDEO_NUM

def save_to_folder(img, folder, name):
    """Save network prediction to folder."""
    check_folder_path(folder)
    full_filename = os.path.join(folder, name)
    # cv2.imwrite assumes the color order of the input image is BGR and saves it in RGB order. 
    cv2.imwrite(full_filename, img) 


def check_folder_path(folder_path):
    """check if folder exists and is a directory. If not, make dir."""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)  # create a directory recursively


def main():
	cam = cv2.VideoCapture(VIDEO_FILE)
	print("FPS: ", cam.get(cv2.CAP_PROP_FPS))
	width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print("Frame width: ", width) # Frame width:  1920.0
	print("Frame height: ", height) # Frame height:  1080.0

	total_num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
	print("Total number of frames: ", total_num_frames)

	res_dim = (int(width/3), int(height/3)) # 640,360

	num_img = 1

	while cam.isOpened() and num_img < total_num_frames:  
		rval, img = cam.read() # BGR order
		if rval:
			img = cv2.resize(img, res_dim, interpolation=cv2.INTER_LANCZOS4)
			# cv2.imshow('Video frames', img)
			save_to_folder(img, FRAME_FOLDER, str(num_img)+'.png')
			num_img = num_img + 1
		else:
		 	break
    
	cam.release()
	# cv2.destroyAllWindows()
	print('Finish!')


if __name__ == "__main__":
    main()
