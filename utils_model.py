import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import utils_data

BATCH_SIZE = 4  # WARNING! Big values can cause out of GPU memory error.
MAX_NUM_CLASSES = 255


def limit_GPU_memory(n_MB):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate n_MB of memory on the first GPU
        try:
            print("Restricting TensorFlow to only allocate ", n_MB, " of memory on the first GPU")
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=n_MB)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def allow_GPU_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)


class ImageBatchGenerator(keras.utils.Sequence):
    """ Helper to iterate over the data (as Numpy arrays).
        imgmask_list - a list of (photo_uri, mask_uri) tuples. """

    def __init__(self, imgmask_list, img_shape=(utils_data.TRAIN_IMG_SIZE, utils_data.TRAIN_IMG_SIZE),
                 batch_size=BATCH_SIZE, one_hot=False, num_classes=MAX_NUM_CLASSES):
        if imgmask_list is None or len(imgmask_list) == 0:
            raise Exception("Empty list of (photo_uri, mask_uri) tuples")
        self.imgmask_list = imgmask_list
        self.image_shape = img_shape
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imgmask_list) // self.batch_size

    def __getitem__(self, idx):
        """ Returns tuple (|photo_1i|, |mask_1i|) of numpy arrays corresponding to batch #idx.
                           |photo_2i|, |mask_2i|
                           |...     |, |...    |
                           |photo_bi|, |mask_bi| """
        i = idx * self.batch_size
        batch_list = self.imgmask_list[i:i+self.batch_size]
        X = np.zeros((self.batch_size,) + self.image_shape + (3,), dtype="float32")  # (3,) is for RGB
        Y = np.zeros((self.batch_size,) + self.image_shape + ((self.num_classes,) if self.one_hot else (1,)), dtype="uint8")
        for j, imgmask in enumerate(batch_list):
            img = utils_data.read_and_pre_process_image(imgmask[0], self.image_shape, normalize=True)
            X[j] = img
            mask, _ = utils_data.read_and_pre_process_mask(imgmask[1], self.image_shape + ((self.num_classes,) if self.one_hot else (1,)), denormalize=True)
            Y[j] = mask
        return X, Y


# Metrics utils
def dice_coef_and_loss(n_classes=MAX_NUM_CLASSES, one_hot=True, smooth=tf.keras.backend.epsilon()):  # TODO add param for background
    def dice_coef(y_true, y_pred):
        """
        Dice metrics. Ignores index 0 (background).
        y_pred is in one-hot format, y_true will be converted, if it is not.
        """
        if one_hot:
            y_true_f = K.cast(K.flatten(y_true[...,1:]), 'float32')
        else:
            y_true_f = K.cast(K.flatten(K.one_hot(y_true, num_classes=n_classes)[..., 1:]), 'float32')
        y_pred_f = K.flatten(y_pred[...,1:])
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))

    def dice_loss(y_true, y_pred):
        """
        Dice loss to minimize.
        """
        return 1 - dice_coef(y_true, y_pred)

    return dice_coef, dice_loss


# Evaluation utils...
