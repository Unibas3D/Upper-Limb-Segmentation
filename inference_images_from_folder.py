import os
import numpy as np
import cv2
import tensorflow as tf
import dataset_colormap
import visualization
import save_predictions
import image_utils
from model import DeepLabModel


def inference(path, model, color_space, output_dir=None, visualize=True):
    """Inferences DeepLab model and visualizes results.

    Args:
      path: a list of image paths.
      model: a loaded DeepLab model.
      color_space: 'rgb' or 'lab', which is based of the training input used.
      output_dir: (optional) path of the folder in which to save the results.
      visualize: (optional) True if you want to visualize results.
    """

    for img_path in path:
        try:
            original_im = cv2.imread(img_path)  # reads in BGR order
        except IOError:
            print('Cannot retrieve image. Please check path: ' + path)
            return

        print('image path: %s' % img_path)

        original_im = image_utils.convert_opencv_input(original_im, color_space) 

        seg_map = model.run(original_im)

        # segmentation colored image
        seg_image = dataset_colormap.label_to_color_image(
            seg_map).astype(np.uint8)

        # if original_im was resized (during model input preprocessing), then restore 
        # the segmentation colored image dimensions to the original_im size 
        if not (original_im.shape[0] == seg_image.shape[0] and original_im.shape[1] == seg_image.shape[1]):
            seg_image = cv2.resize(seg_image,
                                  (original_im.shape[1],
                                  original_im.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)

        if visualize:
            name_without_ext = (os.path.split(img_path)[1]).split('.')[0]
            visualization.plot_im_seg_overlay(original_im, seg_image, seg_map, name_without_ext)

        if not (output_dir is None):
            name = os.path.split(img_path)[1]
            save_predictions.save_to_folder(seg_image, output_dir, name) 
            print('Prediction of image %s saved!' % name)


def main():
    print("TensorFlow version: ", tf.__version__)
    print("TensorFlow is built with cuda: ", tf.test.is_built_with_cuda())
    if tf.test.is_gpu_available():
        print("GPU device: ", tf.test.gpu_device_name())
    else:
        print("No available GPU!")

    model_dir = "deeplab_trained_models"

    DATE = '08_05_21'
    MODEL_NAME = 'model_' + DATE 
    # 'model_13_05_21' # rgb input (resnet_v1_101_beta pretrained on imagenet) 
    # 'model_10_05_21' # rgb input (resnet_v1_50_beta pretrained on imagenet) 
    # 'model_07_05_21' # rgb input (xception-41 pretrained on imagenet)
    # 'model_08_05_21' # rgb input (xception-65 pretrained on imagenet+coco)

    model_path = os.path.join(model_dir, MODEL_NAME)
    print('DeepLab model path: ', model_path)

    MODEL = DeepLabModel(model_path, 360,360) 
    print('Model loaded successfully!')

    IMAGE_DIR = 'test_images/'
    IMAGE_PATH = [os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if x.endswith('.jpg') or x.endswith('.png')]
    OUTPUT_DIR = 'results/' + IMAGE_DIR + '/' + MODEL_NAME
    inference(IMAGE_PATH, MODEL, 'rgb', output_dir=OUTPUT_DIR, visualize=False)


if __name__ == "__main__":
    main()
