#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#get_ipython().run_line_magic('matplotlib', 'inline')

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

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
    
    #RPN_NMS_THRESHOLD = 0.75
    
    LEARNING_RATE = 0.001
    
    DETECTION_MAX_INSTANCES = 400
    MAX_GT_INSTANCES = 400

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 16 roof types

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 3

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
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

class DatasetLoader():
    def __init__(self):
        self.roofs = []
        self.dataset_dir = None
        
    def load_dataset(self, dataset_dir):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, 'map.txt'), 'r') as file:
            lines = file.readlines()
            for line in lines:
                split = line.split()
                self.roofs.append(split)
        
        
    def split_train_val_data(self, _train=.8, _test=.1, _val=.1, SEED=101010):
        if not self.roofs:
            print('Load Dataset before try to split data!')
            return
        files = self.roofs
        count = len(files)
        train_files = self.split_indices(files, _train, SEED)
        validation_files = self.split_indices(files, _val/(len(files)/count), SEED)
        test_files = files
        
        dataset_train = RoofTypeDataset()
        dataset_train.load_roof_data(train_files, self.dataset_dir)
        dataset_train.prepare()
        dataset_val = RoofTypeDataset()
        dataset_val.load_roof_data(validation_files, self.dataset_dir)
        dataset_val.prepare()
        dataset_test = RoofTypeDataset()
        dataset_test.load_roof_data(test_files, self.dataset_dir)
        dataset_test.prepare()

        return (dataset_train, dataset_val, dataset_test)
    
    def split_indices(self, files, split, SEED=101010):
        random.seed(SEED)
        indices = random.sample(range(0, len(files)), int(len(files)*split))
        indices.sort(reverse=True)
        result = []
        for idx in indices:
            result.append(files.pop(idx))
        return result

class RoofTypeDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.roofs = []
        self.types = {
            '1000':1,
            "2100":2,
            "3100":3,
            "3200":4,
            "3300":5,
            "3400":6,
            "3500":7
        }

    def load_roof_data(self, data_list, dataset_dir):
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
        self.add_class("roof_types", 7, "3500")
        
        self.dataset_dir = dataset_dir
        for entry in data_list:
            self.add_image(
                "roof_types",
                image_id=len(self.roofs),
                path=os.path.join(dataset_dir, entry[0])
            )
            self.roofs.append(entry)

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
        img = skimage.io.imread(image_info["path"])
        mask_paths = self.roofs[image_id][1:]
        masks = []
        lbls = np.empty(0).astype(np.int)
        for cnt, mask in enumerate(mask_paths):
            path = os.path.join(self.dataset_dir, mask)
            arr = skimage.io.imread(path).astype(np.bool)
            masks.append(arr)
            lbl = self.types[mask.split('\\')[1]]
            lbls = np.append(lbls, lbl)
        result = np.dstack(np.asarray(masks))
        return result, lbls

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "roof_types":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


dataset_dir = r'C:\Users\MoritzWollenhaupt\Desktop\ArcGIS_Rooftype_Detection\data\bochum\tif\train\512\mrcnn\single_instances_augmented_sobel_min_max_uint16'

loader = DatasetLoader()
loader.load_dataset(dataset_dir)
dataset_train, dataset_val, dataset_test = loader.split_train_val_data()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

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


# ### Augmentation
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seqAug = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 50% of all images
        iaa.LinearContrast((0.75, 1.5)),
        # crop images by -10% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.1, 0.1),
            #pad_mode=ia.ALL,
            pad_cval=0
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #    # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
            rotate=(-175, 175), # rotate by -175 to +175 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbor or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval = 0
            #mode=ia.ALL # use any of scikit-image's warping modes
        ))
    ],
    random_order=True
)


# ## Training

epochs = 400

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/100, 
            epochs=epochs, 
            layers='all',
            #augmentation=seqAug
           )