import os
from Inference import Inference
from Collector import Collector
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
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

    inference = Inference()
    sess = inference.get_model_session(PATH_TO_CKPT)

    # Input tensor is the image.
    image_tensor = inference.get_input_tensor()

    # Output tensors are the detection boxes, scores, and classes.
    detection_boxes = inference.get_output_tensor()

    # Each score represents level of confidence for each of the objects.
    detection_scores = inference.get_model_detection_scores()
    detection_classes = inference.get_model_detection_classes()

    # Number of objects detected.
    num_detections = inference.get_model_detected_objects()

    camera = PiCamera()
    camera.resolution = (
        IM_WIDTH,
        IM_HEIGHT
    )
    camera.framerate = 10
    rawCapture = PiRGBArray(
        camera,
        size=(IM_WIDTH, IM_HEIGHT)
    )
    rawCapture.truncate(0)

    # Initialize frame rate calculation
    collector = Collector()
    frame_rate_calc = collector.frame_rate_calc
    freq = collector.freq
    font = collector.font

    for frame1 in camera.capture_continuous(
        rawCapture,
        format="bgr",
        use_video_port=True
    ):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape:
        # [1, None, None, 3]
        # i.e. a single-column array, where each item in the column
        # has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running
        # the model with the image as input.
        (boxes, scores, classes, num) = sess.run(
            [
                detection_boxes,
                detection_scores,
                detection_classes,
                num_detections
            ],
            feed_dict={image_tensor: frame_expanded}
        )

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(
            frame,
            "FPS: {0:.2f}".format(frame_rate_calc),
            (30, 50),
            font,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        # All the results have been drawn on the frame
        # so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()
