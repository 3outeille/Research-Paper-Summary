import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
from skimage.transform import resize

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def SqueezeExcitation(x, filters_in, filters_expand, se_ratio):
    """
        Squeeze and Excitation phase.
        
        Parameters:
        -x: Tensor, input tensor of conv layer.
        -filters_in: Integer, dimension of the input space.
        -filters_expand: Integer, Dimension of the output space after expansion
        -se_ratio: Float, ratio use to squeeze the input filters.
    """
    filters_se = max(1, int(filters_in * se_ratio))
    
    # Squeeze.
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters_expand))(se)
    
    # Excitation.
    se = Conv2D(filters=filters_se,
                kernel_size=1,
                padding='same',
                kernel_initializer= CONV_KERNEL_INITIALIZER,
                use_bias=True)(se)
    
    se = Activation(tf.nn.swish)(se)
    
    se = Conv2D(filters=filters_expand,
                kernel_size=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer= CONV_KERNEL_INITIALIZER,
                use_bias=True)(se)
    
    # Scale.
    x = multiply([x, se])
    
    return x

def __bottleneck(inputs, filters_in, filters_out, kernel_size, expansion_coef, se_ratio, stride, dropout_rate):
    """
        Basic bottleneck structure.
        
        Parameters:
        -inputs: Tensor, input tensor of conv layer.
        -filters_in: Integer, dimension of the input space.
        -filters_out: Integer, dimension of the output space.
        -kernel_size: Integer or tuple of 2 integers, width and height of filters.
        -expansion_coef: Integer, expansion coefficient.
        -se_ratio: Float, ratio use to squeeze the input filters.
        -stride: Integer or tuple of 2 integers, conv stride.
    """
    # Dimension of the output space after expansion.
    filters_expand = filters_in * expansion_coef
    
    # Expansion phase.
    if expansion_coef != 1:
        x = Conv2D(filters=filters_expand,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                   use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.swish)(x)
    else:
        x = inputs
    
    # Dephtwise conv phase.
    x = DepthwiseConv2D(kernel_size=kernel_size,
                        strides=stride,
                        padding='same',
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(tf.nn.swish)(x)
    
    # Squeeze and Excitation phase.
    x = SqueezeExcitation(x, filters_in, filters_expand, se_ratio)
    
    # Output phase.
    x = Conv2D(filters=filters_out,
               kernel_size=1,
               padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               use_bias=False)(x)
    
    x = BatchNormalization()(x)
    
    # MobileNetV2 paper: "Add skip connection when stride=1."
    if (stride == 1 and filters_in == filters_out):
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = add([x, inputs])
    
    return x

def MBConvBlock(inputs, filters_in, filters_out, kernel_size, expansion_coef, se_ratio, stride, repeat, dropout_rate):
    """
    """
    x = __bottleneck(inputs, filters_in, filters_out, kernel_size, expansion_coef, se_ratio, stride, dropout_rate)
    
    
    # MobileNetV2 paper: "The first layer of eachsequence has a stride s and all others use stride 1"
    filters_in = filters_out
    stride = 1
    
    for i in range(1, repeat):
        x = __bottleneck(x, filters_in, filters_out, kernel_size, expansion_coef, se_ratio, stride, dropout_rate)
        
    return x

def ConvBlock(inputs, filters, kernel_size, stride=1, padding='same'):
    """
    
    """ 
    x = inputs
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=stride,
               padding=padding,
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

    x = BatchNormalization()(x)
    x = Activation(tf.nn.swish)(x)
    
    return x

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


