import os
import numpy as np
import cv2
import tensorflow as tf
import dataset_colormap
import visualization
import save_predictions
import image_utils
from model import DeepLabModel


def inference(cam, model, color_space):
    """Inferences DeepLab model and visualizes results.

    Args:
      cam: VideoCapture objects (reading stream from webcam or video).
      model: a loaded DeepLab model.
      color_space: 'rgb' or 'lab', which is based of the training input used.
    """  
    while cam.isOpened():  
        #print("Cam opened!")
        rval, original_im = cam.read()
        if rval:                
            original_im = image_utils.convert_opencv_input(original_im, color_space)
            
            seg_map = model.run(original_im)
            
            # segmentation colored image
            seg_image = dataset_colormap.label_to_color_image(seg_map).astype(np.uint8)

            # if original_im was resized (during model input preprocessing), then restore 
            # the segmentation colored image dimensions to the original_im size 
            if not (original_im.shape[0] == seg_image.shape[0] and original_im.shape[1] == seg_image.shape[1]):
                seg_image = cv2.resize(seg_image,
                                        (original_im.shape[1],
                                        original_im.shape[0]),
                                        interpolation=cv2.INTER_LANCZOS4)
                  
            visualization.plot_overlay(original_im, seg_image, seg_map)
            
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27: # Exit if 'q' or ESC is pressed (27 is the ascii value of ESC)
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print('Exit!')


def get_cam(video_file=None):
    if video_file is None:
        # webcam
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # change cam ID if necessary (default 0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    else:
        # video file
        if os.path.isfile(video_file):
            cam = cv2.VideoCapture(video_file)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        else:
            print("File %s not exist!" % video_file)
            cam = None
    return cam


def main():
    print("TensorFlow version: ", tf.__version__)
    print("TensorFlow is built with cuda: ", tf.test.is_built_with_cuda())
    if tf.test.is_gpu_available():
        print("GPU device: ", tf.test.gpu_device_name())
    else:
        print("No available GPU!")

    model_dir = "deeplab_trained_models"

    MODEL_NAME = 'model_08_05_21' 
    # 'model_13_05_21' # rgb input (resnet_v1_101_beta pretrained on imagenet) 
    # 'model_10_05_21' # rgb input (resnet_v1_50_beta pretrained on imagenet) 
    # 'model_07_05_21' # rgb input (xception-41 pretrained on imagenet)
    # 'model_08_05_21' # rgb input (xception-65 pretrained on imagenet+coco)

    model_path = os.path.join(model_dir, MODEL_NAME)
    print('DeepLab model path: ', model_path)

    MODEL = DeepLabModel(model_path, 360, 360)  # 640,360 for vasca 
    print('Model loaded successfully!')    

    print("Please, press 'q' or ESC to exit.") 

    ## VIDEO INFERENCE
    # VIDEO_PATH = 'video/vid9.mp4'
    # cam = get_cam(VIDEO_PATH) 

    ## WEBCAM INFERENCE
    cam = get_cam()

    if not (cam is None):
        inference(cam, MODEL, 'rgb')


if __name__ == "__main__":
    main()
