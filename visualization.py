from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import cv2

import dataset_colormap


LABEL_NAMES = np.asarray([
        'background',
        'hand'
    ])
    
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = dataset_colormap.label_to_color_image(FULL_LABEL_MAP)

def plot_im_seg_overlay(image, seg_image, seg_map, name):
    """Visualizes input image, segmentation result and overlay view."""
    fig = plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input image ' + name )

    plt.subplot(grid_spec[1])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('Segmentation result')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('Segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

def plot_overlay(image, seg_image, seg_map):
    """Visualizes an overlay view (input image and segmentation result)."""
    alpha = 0.5
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, alpha, seg_image, 1-alpha, 0.0)
    cv2.imshow('Segmentation overlay', overlay)