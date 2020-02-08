import tensorflow as tf


class Inference:
    """
    Load classification model.
    """
    def __init__(self):
        """
        Initialize Collector model object.
        """
        self.detection_graph = tf.Graph()

    def get_model_session(self, model_path):
        """
        Return model session into TensorFlow object.

        Args:
        ---------
            'model_path': path to frozen detection graph .pb file
        Return:
        ---------
            session as tensorflow object
        """
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=self.detection_graph)

            return sess

    def get_input_tensor(self):
        """
        Returns input tensor of the image

        Return:
        ---------

        """
        return self.detection_graph.get_tensor_by_name('image_tensor:0')

    def get_output_tensor(self):
        """
        Returns output tensor of the image
        Output tensors are the detection boxes, scores, and classes
        Each box represents a part of the image where a particular object
        was detected

        Return:
        ---------

        """
        return self.detection_graph.get_tensor_by_name('detection_boxes:0')

    def get_model_detection_scores(self):
        """
        Each score represents level of confidence for each of the objects.

        Return:
        ---------

        """
        return self.detection_graph.get_tensor_by_name('detection_scores:0')

    def get_model_detection_classes(self):
        """
        The score is shown on the result image, together with the class label.

        Return:
        ---------

        """
        return self.detection_graph.get_tensor_by_name('detection_classes:0')

    def get_model_detected_objects(self):
        """
        Number of objects detected.

        Return:
        ---------

        """
        return self.detection_graph.get_tensor_by_name('num_detections:0')
