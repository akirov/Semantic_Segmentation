import os
import cv2
import numpy as np

IMAGES_FOLDER = "photos"
MASKS_FOLDER = "masks"
TRAIN_IMG_PREFIX = "timg_"
TRAIN_IMG_SIZE = 512


def get_training_images_and_masks(root):
    """ Expected folder structure:
     /root/
     -----/[pref1]_photos/
     --------------------/[img_name11.jpg]
     --------------------/[img_name12.jpg]
     --------------------/...
     -----/[pref1]_masks/
     -------------------/[img_name11].png
     -------------------/[img_name12].png
     -------------------/...
     -----/[pref2]_photos/
     --------------------/[img_name21.jpg]
     --------------------/...
     -----/[pref2]_masks/
     -------------------/[img_name21].png
     -------------------/...
     -----/...
    where strings in [] are arbitrary, but have to match. """
    subdirs = next(os.walk(root))[1]
    image_to_mask = dict()
    for subdir in subdirs:
        if subdir.endswith('_photos'):
            photos_dir = os.path.join(root, subdir)
            masks_dir = photos_dir[:photos_dir.rfind('_photos')] + '_masks'
            if not os.path.exists(masks_dir) or not os.path.isdir(masks_dir):
                print("WARNING: corresponding _masks folder not found for '{}'".format(photos_dir))
                continue
            photos = next(os.walk(photos_dir), (None, None, []))[2]
            masks = set(next(os.walk(masks_dir), (None, None, []))[2])
            if len(photos) == 0 or len(masks) == 0:
                print("WARNING: either '{}' or '{}' folder is empty".format(photos_dir, masks_dir))
                continue
            for photo_ext in photos:
                photo = photo_ext[:photo_ext.rfind('.')]
                mask = photo + '.png'
                if mask in masks:
                    image_to_mask[os.path.join(photos_dir, photo_ext)] = os.path.join(masks_dir, mask)
                else:
                    print("WARNING: can't find .png mask correspondig to '{}' photo".format(photo_ext))
    return image_to_mask


def pre_process_images_and_masks(root, output_folder, output_shape=(TRAIN_IMG_SIZE,TRAIN_IMG_SIZE), overwrite=False):
    images_to_masks = get_training_images_and_masks(root)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_folder = os.path.join(output_folder, IMAGES_FOLDER)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    else:
        print("INFO: '{}' folder already exists!".format(images_folder))

    masks_folder = os.path.join(output_folder, MASKS_FOLDER)
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
    else:
        print("INFO: '{}' folder already exists!".format(masks_folder))

    file_idx = 0
    num_classes = 0
    for image_uri, mask_uri in images_to_masks.items():
        new_image_uri = os.path.join(images_folder, "{0}{1:08d}.png".format(TRAIN_IMG_PREFIX, file_idx))  # png or jpg ???
        if not overwrite and os.path.exists(new_image_uri):
            print("ERROR: '{}' already exists!".format(new_image_uri))
        else:
            scaled_image = read_and_pre_process_image(image_uri, output_shape, False)
            cv2.imwrite(new_image_uri, scaled_image)

        new_mask_uri = os.path.join(masks_folder, "{0}{1:08d}.png".format(TRAIN_IMG_PREFIX, file_idx))  # Same as the photo. Or "{0}{1:08d}_mask.png" ?
        if not overwrite and os.path.exists(new_mask_uri):
            print("ERROR: '{}' already exists!".format(new_mask_uri))
        else:
            scaled_mask, uc = read_and_pre_process_mask(mask_uri, output_shape, True)
            if uc > num_classes:
                num_classes = uc
            cv2.imwrite(new_mask_uri, scaled_mask)

        file_idx += 1

    print("Processed {0} files. Number of different classes: {1}.".format(file_idx, num_classes))
    return num_classes


def read_and_pre_process_image(image_uri, output_shape=(TRAIN_IMG_SIZE,TRAIN_IMG_SIZE), normalize=False):
    image = cv2.imread(image_uri, cv2.IMREAD_COLOR)  # or IMREAD_UNCHANGED ?
    if image is None:
        return None
    if image.shape[:2] != output_shape[:2]:
        image = cv2.resize(image, output_shape[:2], interpolation=cv2.INTER_NEAREST)  # Same as mask. Or use cv2.INTER_AREA ???
    if normalize:
        # Normalize the histogram first?
        image = image / 255.0
    return image


def read_and_pre_process_mask(mask_uri, output_shape=(TRAIN_IMG_SIZE,TRAIN_IMG_SIZE), denormalize=True):
    mask = cv2.imread(mask_uri, cv2.IMREAD_GRAYSCALE)  # Or IMREAD_UNCHANGED and convert manually below?
    if mask is None:
        return None
    if mask.shape[:2] != output_shape[:2]:
        mask = cv2.resize(mask, output_shape[:2], interpolation=cv2.INTER_NEAREST)
    #if mask.ndim >= 3:
    #    u8_mask = np.zeros(mask.shape[:2], np.uint8)
    #    # Manual conversion: consider [RGB] as a 3-bit binary. Thus we will have 8 classes.
    #else:
    unique_colors = np.unique(mask)
    uc = len(unique_colors)
    if denormalize:
        # Convert sparse values to sequential [0..uc)
        if unique_colors[-1] > (uc - 1):
            nc = 0
            for c in unique_colors:
                mask[mask == c] = nc
                nc += 1
    if len(output_shape) > 2:
        if len(output_shape) == 3 and output_shape[2] == 1:
            mask = np.expand_dims(mask, 2)
        else:  # one-hot or error
            print("ERROR! One-hot is not implemented.")
            exit(-1)
    return mask, uc


def convert_gray_mask_to_RGB(mask):
    #if mask.ndim > 2 and mask.shape[2] > 1:  # Already in RGB
    #    return mask
    # Count the number of unique gray levels and map them to distinct colors...
    grays = np.unique(mask)
    if grays.size < 9:
        mapped_colors = [(0,0,0), (0,0,255), (0,255,255), (255,0,0), (255,255,0), (255,0,255), (0,255,0), (255,255,255)][:grays.size]
        mapped_colors[-1] = (255,255,255)
        color_map = dict(zip(grays, mapped_colors))
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                rgb_mask[y][x] = color_map[mask[y][x]]
    else:
        rgb_mask = cv2.cvtColor(mask, cv2.CV_GRAY2RGB)  # Or use PIL.ImageOps.autocontrast(mask) ?
    return rgb_mask


def upsize_mask(mask, w, h):
    scaled_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)  # Or cv2.INTER_CUBIC
    return scaled_mask


def apply_mask_to_image(image, mask):
    # https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv
    return image


def get_timg_file_list(folder):  # Add ext parameter?
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(TRAIN_IMG_PREFIX) and (f.endswith('.png') or f.endswith('.jpg'))]
