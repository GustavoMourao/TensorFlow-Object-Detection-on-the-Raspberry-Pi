import cv2


class Collector:
    """
    Class that manipulates opencv objects (raw image).
    """
    def __init__(self):
        """
        Initialize frame rate calculation.
        """
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
