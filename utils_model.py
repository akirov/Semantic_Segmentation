import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils_data

BATCH_SIZE = 4  # WARNING! Big values can cause our of GPU memory error.


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

    def __init__(self, imgmask_list, img_shape=(utils_data.TRAIN_IMG_SIZE, utils_data.TRAIN_IMG_SIZE), batch_size=BATCH_SIZE):  # TODO add one_hot and num_channels parameters?
        if imgmask_list is None or len(imgmask_list) == 0:
            raise Exception("Empty list of (photo_uri, mask_uri) tuples")
        self.imgmask_list = imgmask_list
        self.image_shape = img_shape
        self.batch_size = batch_size

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
        Y = np.zeros((self.batch_size,) + self.image_shape + (1,), dtype="uint8")  # Or + (num_channels,) for one-hot
        for j, imgmask in enumerate(batch_list):
            img = utils_data.read_and_pre_process_image(imgmask[0], self.image_shape, normalize=True)
            X[j] = img
            mask, _ = utils_data.read_and_pre_process_mask(imgmask[1], self.image_shape + (1,), denormalize=True)
            Y[j] = mask
        return X, Y


# Metrics utils

# Dice loss for several classes?
