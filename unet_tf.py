import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import random
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU, OneHotMeanIoU
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageOps
import utils_data
import utils_model
import cv2

# Use base SegModel class and put all these there?

# TODO Put these in a config file
GPU_MEM_TO_USE_MB = 8192  # 8 GB
EPOCHS = 20
SAVE_MODEL_FOLDER = "saved_models"
SAVE_MODEL_FILE = os.path.join(SAVE_MODEL_FOLDER, "unet_tf.h5")
TRAIN_VAL_SPLIT = 0.80
USE_ONE_HOT = True
USE_DICE_LOSS = False
USE_IOU_LOSS = False
USE_MIOU_METRICS = True
USE_DROPOUT = False
DROP_RATE = 0.50
LR_ADAM = 0.0001


def load_saved_model(saved_model_file=SAVE_MODEL_FILE, **kwargs):
    try:
        saved_model = load_model(saved_model_file, **kwargs)
        return saved_model
    except (ImportError, IOError) as error:
        print(error)
        return None


def create_model(num_classes=utils_model.MAX_NUM_CLASSES, input_shape=(utils_data.TRAIN_IMG_SIZE, utils_data.TRAIN_IMG_SIZE, 3),
                 n_filters=64, use_dropout=USE_DROPOUT):  # Add loss, metrics and network depth params?
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(n_filters, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(n_filters, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(n_filters*2, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(n_filters*2, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    if use_dropout:
        out4 = Dropout(DROP_RATE)(conv4, training=True)
    else:
        out4 = conv4
    pool4 = MaxPool2D(pool_size=(2, 2))(out4)

    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    if use_dropout:
        out5 = Dropout(DROP_RATE)(conv5, training=True)
    else:
        out5 = conv5

    up6 = Conv2D(n_filters*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(out5))  # UpSampling2D is like the opposite of pooling where it repeats rows and columns of the input. Conv2DTranspose performs up-sampling and convolution.
    # BatchNormalization?
    merge6 = concatenate([out4, up6], axis=3)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # BatchNormalization?
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # BatchNormalization?

    up7 = Conv2D(n_filters*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(n_filters*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(n_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)  # softmax -> probability for each class

    model = Model(inputs=inputs, outputs=conv10, name="U-Net")

    if USE_DICE_LOSS:
        # Dice loss
        dice_coef, dice_loss = utils_model.dice_coef_and_loss(num_classes, USE_ONE_HOT)
        model.compile(optimizer=Adam(learning_rate=LR_ADAM),
                      loss=dice_loss,
                      metrics=[dice_coef])
    elif USE_IOU_LOSS:
        # IoU loss
        iou_coef, iou_loss = utils_model.IoU_coef_and_loss(num_classes, USE_ONE_HOT)
        model.compile(optimizer=Adam(learning_rate=LR_ADAM),
                      loss=iou_loss,
                      metrics=[iou_coef])
    elif USE_MIOU_METRICS:
        if USE_ONE_HOT:
            model.compile(optimizer=Adam(learning_rate=LR_ADAM),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=[OneHotMeanIoU(num_classes=num_classes, name='mIOU')])  # ['accuracy', OneHotMeanIoU(...)] ?
        else:
            model.compile(optimizer=Adam(learning_rate=LR_ADAM),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=[MeanIoU(num_classes=num_classes, name='mIOU')])
    else:
        # pixel-wise accuracy is not very good metrics when we have too much background and small areas of other classes
        if USE_ONE_HOT:
            model.compile(optimizer=Adam(learning_rate=LR_ADAM),
                          loss=tf.keras.losses.CategoricalCrossentropy(),  # CategoricalCrossentropy for one-hot labels
                          metrics=['accuracy'])
        else:
            model.compile(optimizer=Adam(learning_rate=LR_ADAM),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # from_logits=True <-- use this if we output classes, not probs
                          metrics=['accuracy'])

    return model


def train(training_data_folder, num_classes, batch_size=utils_model.BATCH_SIZE, epochs=EPOCHS, save_model_uri=SAVE_MODEL_FILE):
    """ Expects a folder containing pre-processed images (photos/timg_XXXXXXXX.png) and masks (masks/timg_XXXXXXXX.png),
        number or output classes, number of images in training batch, training epochs, file URI where to save the model """
    if batch_size is None: batch_size=utils_model.BATCH_SIZE
    if epochs is None: epochs=EPOCHS
    if save_model_uri is None: save_model_uri = SAVE_MODEL_FILE

    # Get training data
    images_folder = os.path.join(training_data_folder, utils_data.IMAGES_FOLDER)
    if not os.path.exists(images_folder):
        print("ERROR: '{}' folder does not exist!".format(images_folder))
        return
    masks_folder = os.path.join(training_data_folder, utils_data.MASKS_FOLDER)
    if not os.path.exists(masks_folder):
        print("ERROR: '{}' folder does not exist!".format(masks_folder))
        return
    images_list = sorted(utils_data.get_timg_file_list(images_folder))
    masks_list = sorted(utils_data.get_timg_file_list(masks_folder))
    if len(images_list) != len(masks_list):  # TODO and filenames differ
        print("ERROR: Training photos and masks don't match")
        return
    imgmask_list = [(images_list[i], masks_list[i]) for i in range(len(images_list))]
    random.shuffle(imgmask_list)

    # Train/test split
    split_mark = int(len(images_list)*TRAIN_VAL_SPLIT)  # for training and validation
    train_imgmask_list = imgmask_list[:split_mark]
    test_imgmask_list = imgmask_list[split_mark:]

    # Create batch generators. Or we can use tf.data.Dataset.from_tensor_slices((X,y))
    train_gen = utils_model.ImageBatchGenerator(train_imgmask_list, (utils_data.TRAIN_IMG_SIZE, utils_data.TRAIN_IMG_SIZE),\
                                                batch_size, USE_ONE_HOT, num_classes)
    val_gen = utils_model.ImageBatchGenerator(test_imgmask_list, (utils_data.TRAIN_IMG_SIZE, utils_data.TRAIN_IMG_SIZE),\
                                              batch_size, USE_ONE_HOT, num_classes)

    # Setup GPU
    #utils_model.limit_GPU_memory(GPU_MEM_TO_USE_MB)
    utils_model.allow_GPU_memory_growth()

    # Create the model
    model = create_model(num_classes, n_filters=64)
    model.summary()
    #keras.utils.plot_model(model, to_file="unet_tf.png", show_shapes=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(save_model_uri, save_best_only=True)
        #, EarlyStopping(patience=5, verbose=1)
        #, ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6)
    ]

    # Train the model
    training_hist = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    # Display learning curves
    if USE_DICE_LOSS:
        acc = 'dice_coef'
        val_acc = 'val_dice_coef'
        acc_title = 'Dice metrics'
        acc_axis = 'Dice coef'
        loss_title = 'Dice loss'
    elif USE_IOU_LOSS:
        acc = 'iou_coef'
        val_acc = 'val_iou_coef'
        acc_title = 'IoU metrics'
        acc_axis = 'IoU coef'
        loss_title = 'IoU loss'
    elif USE_MIOU_METRICS:
        acc = 'mIOU'
        val_acc = 'val_mIOU'
        acc_title = 'mIoU metrics'
        acc_axis = 'mIoU'
        loss_title = 'mIoU loss'
    else:
        acc = 'accuracy'
        val_acc = 'val_accuracy'
        acc_title = 'Model accuracy'
        acc_axis = 'accuracy'
        loss_title = 'Model loss'
    plt.plot(training_hist.history[acc])
    plt.plot(training_hist.history[val_acc])
    plt.title(acc_title)
    plt.ylabel(acc_axis)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(training_hist.history['loss'])
    plt.plot(training_hist.history['val_loss'])
    plt.title(loss_title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Evaluate: infer on test data, confusion matrices, ... Extract this in a separate function.


def infer(input_images, output_folder = ".", saved_model=SAVE_MODEL_FILE):
    #utils_model.limit_GPU_memory(GPU_MEM_TO_USE_MB)
    if USE_DICE_LOSS:
        dice_coef, dice_loss = utils_model.dice_coef_and_loss()  # TODO pass num_classes and USE_ONE_HOT here! Or do not save/load the loss?
        model = load_saved_model(saved_model, custom_objects={dice_loss.__name__: dice_loss, dice_coef.__name__: dice_coef})
    elif USE_IOU_LOSS:
        #iou_coef, iou_loss = utils_model.IoU_coef_and_loss()
        model = load_saved_model(saved_model, compile=False)  # compile=False is to prevent complaining about missing loss
                                 #custom_objects={iou_loss.__name__: iou_loss, iou_coef.__name__: iou_coef})  # don't need these for inference
    else:
        model = load_saved_model(saved_model)
    if model is None:
        print("ERROR loading {} model".format(saved_model))
        return
    model.summary()
    for img_uri in input_images:
        # TODO Use sliding window instead of scaling?
        img = utils_data.read_and_pre_process_image(img_uri, normalize=True)
        if img is None:
            continue
        # Or create a generator object with batch_size = 1 ?
        mask = forward_pass(img, model)
        mask_rgb = utils_data.convert_gray_mask_to_RGB(mask)
        # dilate?
        #cv2.imshow("RGB mask", mask_rgb)
        #cv2.waitKey(0)
        image = cv2.imread(img_uri, cv2.IMREAD_UNCHANGED)  # or IMREAD_COLOR
        pred_mask = utils_data.upsize_mask(mask_rgb, image.shape[1], image.shape[0])
        # Apply mask on image?
        img_file_name = os.path.basename(img_uri)
        result_file = img_file_name[:img_file_name.rfind('.')] + "_mask.png"
        cv2.imwrite(os.path.join(output_folder, result_file), pred_mask)


def forward_pass(image, model):
    img = np.expand_dims(image, 0)
    mask = model.predict(img)[0]
    mask = np.argmax(mask, axis=-1)
    #disp_mask = ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(np.expand_dims(mask, axis=-1)))
    #disp_mask.show()
    return mask.astype(np.uint8)
