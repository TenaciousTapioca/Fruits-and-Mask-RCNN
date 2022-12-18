"""
Obtain trained weights on fruits dataset using COCO pretrained weights
# Spyder Console:     runfile('directory/to/fruits_MaskRCNN/train.py')
    -To show mask of an image (change image_id in main): 
                      runfile('directory/to/fruits_MaskRCNN/train.py', args='showMask')
# Otherwise:          python3 train.py
    -Show image mask: python3 train.py showMask
"""

import os
import sys
import json
import numpy as np
import skimage.draw
import imgaug.augmenters # Image augmentations
from mrcnn import visualize

# Enable memory growth for GPUs to hopefully avoid Out of Memory error
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Import Mask RCNN
ROOT_DIR = os.path.abspath("./") # Root directory of the project (fruits_MaskRCNN/)
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained COCO weights file (in root directory of project [fruits_MaskRCNN/])
# our training weights will be built ontop of this
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints (.h5)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory of image dataset to train and validate upon
DATASET_DIR = os.path.join(ROOT_DIR, "fruits")
# Directory of images to apply prediction on for inference.py
IMAGE_DIR = os.path.join(ROOT_DIR, "inferenceImages")

# Configurations: changing default values for predicting
class FruitsConfig(Config):
    # Configuration name
    NAME = "fruits"

    # BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # 1 Background + 3 fruits [apple, orange, banana]

    # Number of training steps per epoch
    # Each epoch creates an .h5 weight file (~0001.h5, ~0002.h5, . . .)
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
# print updated configs
#config = FruitsConfig()
#config.display()

# Fruit Dataset specifications: classes, masks, and image reference
class FruitsDataset(utils.Dataset):
    def load_fruits(self, dataset_dir, subset):
        # Add classes in the form (source, id, class); fruits dataset only contains 3 classes
        self.add_class("fruits", 1, "apple")
        self.add_class("fruits", 2, "orange")
        self.add_class("fruits", 3, "banana")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            
            # Obtain class ids for each class of the fruits dataset
            fruits = [s['region_attributes']['label'] for s in a['regions'].values()]
            fruit_dict = {"apple": 1, "orange": 2, "banana": 3}
            class_ids = [fruit_dict[i] for i in fruits]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)   # image as multidimensional Numpy array
            height, width = image.shape[:2]         # shape[0] = rows = height; shape[1] = columns = width
                                                    # slicing: take elements [0, 2)

            self.add_image(
                "fruits",
                image_id=a['filename'], # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    # Obtain masks for an image from its polygon annotations
    def load_mask(self, image_id):
        # If not a fruits dataset image, delegate to parent class.
        # and return an empty mask
        info = self.image_info[image_id]
        if info["source"] != "fruits":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        class_ids = info["class_ids"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            
            # Helps avoid IndexError: index # is out of bounds for axis 0 with size #
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            
            mask[rr, cc, i] = 1
        
        # Return mask and array of class IDs of each instance
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids
    
    # Image path
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# Training
def train(model):
    # Load training dataset
    dataset_train = FruitsDataset()
    dataset_train.load_fruits(DATASET_DIR, "train")
    dataset_train.prepare()

    # Load validation dataset
    dataset_val = FruitsDataset()
    dataset_val.load_fruits(DATASET_DIR, "val")
    dataset_val.prepare()

    # Training only the head layers
    # Apply image augmentation (flipping and/or scaling) to add diversity to image dataset
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation = imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.OneOf([
                imgaug.augmenters.Fliplr(0.5),                   # probability = 0.5
                imgaug.augmenters.Affine(scale=(0.5, 1.5))])))   # scale to 50% to 150% of size

# The following is not run when imported
if __name__ == '__main__':
    
    # Train model
    if len(sys.argv) == 1:
        config = FruitsConfig()
        config.display()
        # training initially with COCO weights
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
        if not os.path.exists(COCO_WEIGHTS_PATH):
            print("mask_rcnn_coco.h5 not found in the weights_path:{}\nDownloading COCO weights...".format(COCO_WEIGHTS_PATH))
            utils.download_trained_weights(COCO_WEIGHTS_PATH)

        # Begin training model; checkpoint weights (.h5) are in the log path
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
                            "mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])
        train(model)
    
    # Visualize the mask for an image given its image_id
    elif sys.argv[1] == "showMask":
        # Visualize the mask for an image given its image_id
        dataset = FruitsDataset()
        dataset.load_fruits("fruits", "train")
        dataset.prepare()

        #image_id = 152
        image_id = 181
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image=image, mask=mask, class_ids=class_ids, class_names=dataset.class_names)