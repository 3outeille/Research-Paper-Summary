import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from skimage.transform import resize

def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]

# For the exact scaling technique we follow the official implementation as the paper does not tell us.
# https://github.com/tensorflow/tpu/blob/01574500090fa9c011cb8418c61d442286720211/models/official/efficientnet/efficientnet_model.py#L101-L125

def scaled_repeats(n, d_coef):
    return int(math.ceil(n * d_coef))

# Snap number of channels to multiple of 8 for optimized implementations
def scaled_channels(n, w_coef):
    n = n * w_coef
    m = max(8, int(n + 8 / 2) // 8 * 8)

    if m < 0.9 * n:
        m = m + 8

    return int(m)


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    
    MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
    }
    
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int(
        (image_size / (image_size + crop_padding)) * min(h, w)
    )
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[
                 offset_height: padded_center_crop_size + offset_height,
                 offset_width: padded_center_crop_size + offset_width,
                 ]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image