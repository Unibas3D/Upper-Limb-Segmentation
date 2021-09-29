import os
import cv2

def save_to_folder(prediction, folder, name):
    """Save network prediction to folder."""
    check_folder_path(folder)
    full_filename = os.path.join(folder, name)
    # cv2.imwrite assumes the color order of the input image is BGR and saves it in RGB order. 
    # Since prediction is already in RGB, change to BGR to save it correctly.
    cv2.imwrite(full_filename, cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)) 

def check_folder_path(folder_path):
    """Check if folder exists and is a directory. If not, make dir."""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)  # create a directory recursively