import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class RoofTypeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "roof_types"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class RoofTypeDataset(utils.Dataset):
    
    def __init__(self):
        super().__init__()
        self.roofs = []
        self.types = {
            '0':1,
            '1000':2,
            "2100":3,
            "2200":4,
            "3100":5,
            "3200":6,
            "3300":7,
            "3400":8,
            "3500":9,
            "3600":10,
            "3700":11,
            "3800":12,
            "3900":13,
            "4000":14,
            "5000":15,
            "9999":16
        }

    def load_roof_data(self, dataset_dir, subset):
        """Load a subset of the RoofType dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("roof_types", 1, "0")
        self.add_class("roof_types", 2, "1000")
        self.add_class("roof_types", 3, "2100")
        self.add_class("roof_types", 4, "2200")
        self.add_class("roof_types", 5, "3100")
        self.add_class("roof_types", 6, "3200")
        self.add_class("roof_types", 7, "3300")
        self.add_class("roof_types", 8, "3400")
        self.add_class("roof_types", 9, "3500")
        self.add_class("roof_types", 10, "3600")
        self.add_class("roof_types", 11, "3700")
        self.add_class("roof_types", 12, "3800")
        self.add_class("roof_types", 13, "3900")
        self.add_class("roof_types", 14, "4000")
        self.add_class("roof_types", 15, "5000")
        self.add_class("roof_types", 16, "9999")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        self.dataset_dir = os.path.join(dataset_dir, subset)
        
        with open(os.path.join(self.dataset_dir, 'map.txt'), 'r') as file:
            lines = file.readlines()
            for line in lines:
                split = line.split()
                self.add_image(
                    "roof_types",
                    image_id=len(self.roofs),
                    path=os.path.join(dataset_dir, split[0])
                )
                self.roofs.append(split)

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
    