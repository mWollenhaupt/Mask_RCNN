#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
from imgaug import augmenters as iaa
import imgaug as ia

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class RoofTypeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "roof_types"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
    
    #RPN_NMS_THRESHOLD = 0.8
    
    LEARNING_RATE = 0.001
    
    DETECTION_MAX_INSTANCES = 400
    MAX_GT_INSTANCES = 400

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 16 roof types

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 3

    # Use smaller anchors because our image and objects are small
    BACKBONE_STRIDES = (4, 8, 16, 32, 64, 128, 256)
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 256, 384)  # anchor side in pixels
    
    #RPN_ANCHOR_SCALES = (10, 20, 40, 80, 160)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 300

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 15
    
config = RoofTypeConfig()
config.display()


class RoofTypeDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.roofs = []
        self.types = {
            '1000':1,
            '2100':2,
            '3100':3,
            '3200':4,
            '3300':5,
            '3400':6
        }

    def load_roof_data(self, dataset_dir):
        """Load a subset of the RoofType dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Add classes
        self.add_class("roof_types", 1, "1000")
        self.add_class("roof_types", 2, "2100")
        self.add_class("roof_types", 3, "3100")
        self.add_class("roof_types", 4, "3200")
        self.add_class("roof_types", 5, "3300")
        self.add_class("roof_types", 6, "3400")
        
        
        self.dataset_dir = dataset_dir
        for image_set in os.listdir(self.dataset_dir):
            image = os.path.join(self.dataset_dir, image_set, '{}.tif'.format(image_set))
            self.add_image(
                "roof_types",
                image_id=os.path.split(image)[0],
                path=image
            )

    def load_mask(self, image_id):
        """Load instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "roof_types":
            return super(self.__class__, self).load_mask(image_id)

        mask_arrs = []
        mask_lbls = np.empty(0).astype(np.int)
        image_id = image_info['id']
        mask_dirs = [os.path.join(image_id, md) for md in os.listdir(image_id) if os.path.isdir(os.path.join(image_id, md))]
        for mask_dir in mask_dirs:
            mask_type = os.path.split(mask_dir)[1]
            for file in os.listdir(mask_dir):
                arr = skimage.io.imread(os.path.join(mask_dir, file)).astype(np.bool)
                mask_arrs.append(arr)
                mask_lbls = np.append(mask_lbls, self.types[mask_type])
        mask_stack = np.dstack(np.asarray(mask_arrs))
        return mask_stack, mask_lbls
        

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "roof_types":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



#path_train = r'C:\Users\MoritzWollenhaupt\Desktop\ArcGIS_Rooftype_Detection\data\bochum_koeln\tif\chips_single_masks_sobel_global\train'
#path_val = r'C:\Users\MoritzWollenhaupt\Desktop\ArcGIS_Rooftype_Detection\data\bochum_koeln\tif\chips_single_masks_sobel_global\val'
#path_test = r'C:\Users\MoritzWollenhaupt\Desktop\ArcGIS_Rooftype_Detection\data\bochum_koeln\tif\chips_single_masks_sobel_global\test'

path_train = r'C:\Users\MoritzWollenhaupt\Desktop\chips_single_masks_v2_enrich\train'
path_val = r'C:\Users\MoritzWollenhaupt\Desktop\chips_single_masks_v2_enrich\val'
path_test = r'C:\Users\MoritzWollenhaupt\Desktop\chips_single_masks_v2_enrich\test'



dataset_train = RoofTypeDataset()
dataset_train.load_roof_data(path_train)
dataset_train.prepare()

dataset_val = RoofTypeDataset()
dataset_val.load_roof_data(path_val)
dataset_val.prepare()

dataset_test = RoofTypeDataset()
dataset_test.load_roof_data(path_test)
dataset_test.prepare()

# ## Create Model

# In[8]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[9]:


# Which weights to start with?
#init_with = "coco"  # imagenet, coco, or last
init_with = "last"
#init_with = "imagenet"


if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training

# In[ ]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
epochs = 350

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/100, 
            epochs=epochs, 
            layers='all',
           )
