import numpy as np
import tensorflow as tf

########################
#### loss functions ####
########################


def dice_loss(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2.0 * intersection) / (
        tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred)) + smooth
    )


def __gaussuian_kernel_4D(kernel_size, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    _gauss = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return _gauss[..., None, None] / _gauss.sum()


def SSIM_loss(y_true, y_pred):
    c1 = 1.0
    c2 = 1.0
    p = tf.expand_dims(y_true, axis=-1)  # 4D-tensor, shape=(1,512,512,1) for Conv2D
    q = tf.expand_dims(y_pred, axis=-1)
    ker = __gaussuian_kernel_4D(kernel_size=11, sigma=1.5)
    mu_p = tf.nn.conv2d(p, ker, strides=1, padding="VALID")
    mu_q = tf.nn.conv2d(q, ker, strides=1, padding="VALID")
    mu2_p = tf.square(mu_p)
    mu2_q = tf.square(mu_q)
    mu_pq = tf.multiply(mu_p, mu_q)
    sigma2_p = tf.nn.conv2d(tf.square(p), ker, strides=1, padding="VALID") - mu2_p
    sigma2_q = tf.nn.conv2d(tf.square(q), ker, strides=1, padding="VALID") - mu2_q
    sigma_pq = tf.nn.conv2d(tf.multiply(p, q), ker, strides=1, padding="VALID") - mu_pq
    return 1.0 - tf.reduce_mean(
        (2.0 * mu_pq + c1)
        * (2.0 * sigma_pq + c2)
        / ((mu2_p + mu2_q + c1) * (sigma2_p + sigma2_q + c2))
    )

def contour_closure_loss(y_true, y_pred, alpha=1.0, beta=1.0):
    # Ensure y_pred is 4D (batch size, height, width, channels)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    # Compute the gradient of the predicted probabilities using Sobel edges
    grad_y_pred = tf.image.sobel_edges(y_pred)
    grad_mag = tf.sqrt(tf.reduce_sum(tf.square(grad_y_pred), axis=-1))

    # Creating a binary version of y_pred for closure assessment
    binary_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)

    # Use a custom erosion function
    def custom_erosion(image):
        # Create a morphological kernel, a square of ones
        kernel_size = 5
        kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=tf.float32)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)  # Add batch dimension if missing

        # Apply convolution to perform erosion-like operation
        eroded_image = tf.nn.depthwise_conv2d(
            input=image,
            filter=kernel,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

        # Normalize and threshold to mimic erosion
        threshold = kernel_size * kernel_size
        eroded_image = tf.where(eroded_image >= threshold, x=tf.ones_like(eroded_image), y=tf.zeros_like(eroded_image))

        return tf.squeeze(eroded_image, axis=0)  # Remove batch dimension

    # Apply erosion to each image in the batch
    eroded = tf.map_fn(custom_erosion, binary_pred, dtype=tf.float32)

    # Calculate closure penalty
    closure_penalty = -tf.reduce_mean(binary_pred - eroded)

    # Combine the losses
    return alpha * tf.reduce_mean(grad_mag) + beta * closure_penalty



def Dice_SSIM_loss(y_true, y_pred, SSMI_Weight=1.0):
    return dice_loss(y_true, y_pred, smooth=1.0) + SSMI_Weight * SSIM_loss(
        y_true, y_pred
    )

def Dice_SSIM_Closure_loss(y_true, y_pred, SSMI_Weight=1.0,lambda_closure=0.5):
    return dice_loss(y_true, y_pred, smooth=1.0) + SSMI_Weight * SSIM_loss(
        y_true, y_pred) + lambda_closure * contour_closure_loss(y_true, y_pred)