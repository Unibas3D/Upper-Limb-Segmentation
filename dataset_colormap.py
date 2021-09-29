import numpy as np

def create_label_colormap():
    """Creates a label colormap used in EGO_HAND_SEG dataset.

    Returns:
        A colormap for visualizing segmentation results.
    """
    return np.asarray([
        [0, 0, 0],  # black
        [255, 6, 51] # torch red
        # [255, 255, 255]  # white
    ])


def label_to_color_image(label):
    """Adds color defined by the colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]