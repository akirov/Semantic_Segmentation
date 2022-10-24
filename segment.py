import argparse
import utils_data
from utils_model import BATCH_SIZE


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python segment.py")
    subparsers = parser.add_subparsers(help="Sub-commands help", dest="command")

    parser_pre_process = subparsers.add_parser("pre_process", help="Pre-process labeled data to make it suitable for training")
    parser_pre_process.add_argument(
        "--source", "-s", required=True, help="Labeled data root folder, containing SOME_photos and SOME_masks subfolders"
    )
    parser_pre_process.add_argument(
        "--destination", "-d", required=True, help="Folder where to put pre-processed training photos and masks"
    )
    parser_pre_process.add_argument(
        "--width", "-w", type=int, default=utils_data.TRAIN_IMG_SIZE, help="Output images and masks width (and height)"
    )
    parser_pre_process.add_argument(
        "--overwrite", "-o", action='store_true', help="Overwrite destination files"
    )

    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument(
        "--model", "-m", required=True, help="Architecture to use (unet,...)"
    )
    parser_train.add_argument(
        "--training_data", "-d", required=True, help="Folder containing photos/timg_XXXXXXXX.png and masks/timg_XXXXXXXX.png"
    )
    parser_train.add_argument(
        "--num_classes", "-n", type=int, required=True, help="Number of output classes"
    )
    parser_train.add_argument(
        "--batch_size", "-b", type=int, default=BATCH_SIZE, help="Batch size"
    )
    parser_train.add_argument(
        "--epochs", "-e", type=int, default=None, help="Batch size"
    )
    parser_train.add_argument(
        "--save_model", "-s", type=str, default=None, help="Save the model in a file URI"
    )

    parser_infer = subparsers.add_parser("infer", help="Use a model for inference")
    parser_infer.add_argument(
        "--model", "-m", required=True, help="Architecture to use (unet,...)"
    )
    parser_infer.add_argument(
        "--input_images", "-i", nargs='*', required=True, help="Input image URI(s)"
    )
    parser_infer.add_argument(
        "--output_folder", "-o", nargs='?', type=str, default=".", const=".", help="Folder to store the result"
    )
    parser_infer.add_argument(
        "--saved_model", "-s", type=str, default=None, help="Saved model file URI"
    )

    # Add "--evaluate" option?

    args = parser.parse_args()
    #print(args)
    config = vars(args)
    #print(config)

    if args.command == "pre_process":
        utils_data.pre_process_images_and_masks(args.source, args.destination, (args.width,args.width), args.overwrite)

    elif args.command == "train":
        if args.model == 'unet':
            import unet_tf
            unet_tf.train(args.training_data, args.num_classes, args.batch_size,
                          args.epochs if args.epochs is not None else unet_tf.EPOCHS,
                          args.save_model if args.save_model is not None else unet_tf.SAVE_MODEL_FILE)
        else:
            print("Unsupported architecture")

    elif args.command == "infer":
        if args.model == 'unet':
            import unet_tf
            unet_tf.infer(args.input_images, args.output_folder,
                          args.saved_model if args.saved_model is not None else unet_tf.SAVE_MODEL_FILE)
        else:
            print("Unsupported architecture")

    else:
        print("Unknown command : ", args.command)
        print("Try 'segment.py --help'")
        exit()
