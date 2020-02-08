import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util


if __name__ == "__main__":
    """
    Picamera Object Detection using TensorFlow Classifier.

    Edited from first version implemented by Evan Juras.
    Updated date: 02/08/20

    This program uses a TensorFlow classifier to perform object detection.
    It loads the classifier uses it to perform object detection on a Picamera
    feed.
    It draws boxes and scores around the objects of interest in each frame from
    the Picamera.

    Next step aims to cover the same implementation based on
    Openvino framework.
    """
    # Set up camera constants.
    # IM_WIDTH = 1280
    # IM_HEIGHT = 720
    IM_WIDTH = 640
    IM_HEIGHT = 480

    # Name of the directory containing the object detection module we're using.
    # In this case, the model used is Single Shot MultiBox Detector (SSD).
    # Similar of YOLO.
    # The model was trained with Common Objects in Context (Coco) data set.
    # Coco contains labeled images of 91 categories of objects such as people,
    # umbrellas and cars.
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

    # Grab path to current working directory.
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains
    # the model that is used for object detection.
    PATH_TO_CKPT = os.path.join(
        CWD_PATH,
        MODEL_NAME,
        'frozen_inference_graph.pb'
    )

    # Path to label map file.
    PATH_TO_LABELS = os.path.join(
        CWD_PATH,
        'data',
        'mscoco_label_map.pbtxt'
    )

    # Number of classes the object detector can identify.
    NUM_CLASSES = 90

    # Load the label map.
    # Label maps map indices to category names, so that when the convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=NUM_CLASSES,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
