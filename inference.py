"""
Use trained weights on fruits dataset to predict on target image
# Spyder Console: runfile('directory/to/fruits_MaskRCNN/inference.py', args='fruits2.jpg')
# Otherwise:      python3 inference.py fruits2.jpg
"""

import os
import sys
import skimage.io
import tensorflow as tf
import matplotlib.pyplot as plt
from mrcnn import visualize
import mrcnn.model as modellib
import train # Import directories and FruitsConfig/Dataset

if __name__ == '__main__':
    # Ask for image name in IMAGE_DIR; raise error if nonexistent
    if not len(sys.argv) == 2:
        raise Exception("Specify target image to predict on (inference.py image.jpg).")
    elif not (os.path.isfile(os.path.join(train.IMAGE_DIR, sys.argv[1]))):
        raise Exception("Target image ({}) could not be found in {}.".format(sys.argv[1], train.IMAGE_DIR))
    
    ROOT_DIR = os.path.abspath("./")            # fruits_MaskRCNN/ -- root dir of project
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # fruits_MaskRCNN/logs -- dir of training weights
    
    # Configurations: changing training values for inferencing
    config = train.FruitsConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time; BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()
    
    # Adjust axes for plot where output image is displayed  on
    def get_ax(rows=1, cols=1, size=16):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    
    # Load validation set
    dataset = train.FruitsDataset()
    dataset.load_fruits(train.DATASET_DIR, "val")
    dataset.prepare()
    
    # Create model in inference mode
    DEVICE = "/gpu:0" # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        
    # After training (for 10 Epochs), use the last training weights in '/logs' (mask_rcnn_fruits_0010.h5)
    MODEL_WEIGHTS_PATH = model.find_last()
    print("\nLoading weights ", MODEL_WEIGHTS_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)
    
    # Read local image as multidimensional Numpy array and run object detection
    image = skimage.io.imread(os.path.join(train.IMAGE_DIR, sys.argv[1]))
    results = model.detect([image], verbose=1)
    
    # Display results
    ax = get_ax(1)
    r = results[0]
    print("\nPrediction Results for {}\nRunning on {} dataset\n".format(sys.argv[1], train.DATASET_DIR))
    print("\nTarget Image: {}\nClasses: {}\nEntities Found: {}".format(sys.argv[1],
                                dataset.class_names, len(r['rois'])))
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    
    """
    # Uncomment to see image augments (horizontal flips and scales) of images in inferenceImages
    import imgaug.augmenters as iaa
    import cv2
    import os
    
    images = []
    for img_path in os.listdir("inferenceImages"):
        path = "inferenceImages\\{}".format(img_path)
        print(path)
        if not os.path.exists(path):
            raise Exception("Not legal path")
        img = cv2.imread(path)
        images.append(img)
    
    # flips at 50% chance; scales from 50% to 150%
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5), 
        iaa.Affine(scale=(0.5, 1.5))
    ])
    
    augmented_images = augmentation(images=images)
    for img in augmented_images:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    """