from CarFinder.features import Features
import cv2
from CarFinder.classifier import Classifier
from CarFinder.hogsubsampler import HogSubSampler
from CarFinder.averager import Averager
from scipy.ndimage.measurements import label
import numpy as np
from CarFinder.utils import pip
from multiprocessing import Pool
import matplotlib.pyplot as plt

class CarFinder:
    def __init__(self, img_size=(1280, 720)):
        self.ft = Features()
        self.clf = Classifier()
        self.clf2 = Classifier()
        self.clf3 = Classifier()
        self.clf4 = Classifier()
        self.image_size = img_size

        self.hoggs = [HogSubSampler(classifier=self.clf.classifier,
                                    features=self.ft,
                                    scaler=self.clf.scaler,
                                    ystart=390, ystop=582,
                                    scale=1,
                                    img_size=img_size),

                      HogSubSampler(classifier=self.clf2.classifier,
                                    features=self.ft,
                                    scaler=self.clf2.scaler,
                                    ystart=400, ystop=592,
                                    scale=1,
                                    img_size=img_size),
                      HogSubSampler(classifier=self.clf3.classifier,
                                    features=self.ft,
                                    scaler=self.clf3.scaler,
                                    ystart=370, ystop=626,
                                    scale=2,
                                    img_size=img_size),
                      HogSubSampler(classifier=self.clf4.classifier,
                                    features=self.ft,
                                    scaler=self.clf4.scaler,
                                    ystart=380, ystop=636,
                                    scale=2,
                                    img_size=img_size)]

        # Heat map thresholding value
        self.threshold = 3
        # Bounding boxes of cars
        self.bboxes = None
        # Unthresholded heatmap
        self.heatmap_raw = None
        # Averaged heatmap
        self.heatmap_averaged = self.__init_heatmap(img_size)
        # Thresholded heatmap
        self.heatmap_threshold = None
        # Averaged Heatmap

        # Labelized image. each number corresponds different car, 0 means no car
        self.labels = None

    def __init_heatmap(self, image_size):
        """
        Initializes heatmap averager.
        
        :param image_size: (x, y) 
        :return: None
        """
        heatmap  = Averager(10, np.zeros((self.image_size[1], self.image_size[0]),dtype=np.float64), True)
        return heatmap

    def pool_wrapper_hog_find(self, hog_idx, image):
        """Wrapper for Pool worker to enable multiprocessing. 
        Currently this is not used."""
        print(self, hog_idx)
        self.hoggs[hog_idx].find(image)
        return True

    def find(self, image):
        """
        This method finds cars from video stream.
         
        :param image: single frame. dtype=np.uint8 and colorspace is BGR.
        :return: None
        """

        heatmap_temp = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float64)
        # Loop through all sliding window searches.
        for hog in self.hoggs:
            heatmap_temp += hog.find(image)

        self.heatmap_raw  = heatmap_temp
        # Average heatmap
        # Put blurred version of raw heatmap to mean filter.
        self.heatmap_averaged.put(cv2.GaussianBlur(self.heatmap_raw.copy(), (41, 41), 0))
        # Threshold by setting all pixels below threshold limit to zero.
        self.heatmap_threshold = self.heatmap_averaged.mean().copy()
        self.heatmap_threshold[self.heatmap_threshold <= self.threshold] = 0
        # Generate labels and then bounding boxes
        self.labels = label(self.heatmap_threshold.copy())
        self.bboxes = self.labels_to_bboxes(self.labels)

    @staticmethod
    def labels_to_bboxes(labels):
        """
        This method converts label image to bounding boxes.
        
        :param labels: 
        :return: bounding boxes 
        """
        # https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/de41bff0-ad52-493f-8ef4-5506a279b812
        # Iterate through all detected cars
        bboxes = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        # Return the image
        return bboxes

    def draw_bounding_boxes(self, image=None, color=(0, 0, 255), thickness=3):
        """
        This method draws bounding boxes found from last frame.
        :param image: 
        :param color: 
        :param thickness: 
        :return: 
        """
        if image is None:
            shape = (self.image_size[1], self.image_size[0], 3)
            image = np.zeros(shape, dtype=np.uint8)

        for p in self.bboxes:
            cv2.rectangle(image, tuple(p[0]), tuple(p[1]),
                          color=color, thickness=thickness)
        return image

    def pip_heatmap_per_frame(self, src, position, size):
        """
        Adds pip image of per-frame-heatmap to src image.
        
        :param src: Original image to where heatmap is added.
        :param pos: Position of top left corner of subimage in image (x, y)
        :param size: Size of the subimage (It'll be resized to this size) (x, y)
        :return: combined image
        """
        # Draw per frame heatmap
        heatmap = self.heatmap_raw.copy().astype(dtype=np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT) * 5
        src = pip(src, heatmap, position, size, border=5,
                  title="Per Frame Heatmap")
        return src

    def pip_heatmap_averaged(self, src, position, size):
        """
        Adds pip image of  averaged heatmap to src image.

        :param src: Original image to where heatmap is added.
        :param pos: Position of top left corner of subimage in image (x, y)
        :param size: Size of the subimage (It'll be resized to this size) (x, y)
        :return: combined image
        """
        # draw averaged heatmap
        heatmap = self.heatmap_averaged.mean().astype(dtype=np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR) * 5
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        src = pip(src, heatmap, position, size, border=5,
                  title="Average Heatmap")
        return src

    def pip_heatmap_threshold(self, src, position, size):
        """
        Adds pip image of heatmap threshold to src image.

        :param src: Original image to where heatmap is added.
        :param pos: Position of top left corner of subimage in image (x, y)
        :param size: Size of the subimage (It'll be resized to this size) (x, y)
        :return: combined image
        """
        heatmap = self.heatmap_threshold.copy().astype(dtype=np.uint8)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR) * 5
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        src = pip(src, heatmap, position, size, border=5,
                  title="Heatmap Threshold")
        return src

    def pip_labels(self, src, position, size):
        """
        Adds pip image of labels to src image.

        :param src: Original image to where labels image is added.
        :param pos: Position of top left corner of subimage in image (x, y)
        :param size: Size of the subimage (It'll be resized to this size) (x, y)
        :return: combined image
        """
        # Idiotic way to make labels visible
        labels = carfinder.labels[0].copy().astype(dtype=np.uint8) * 20
        labels = cv2.cvtColor(labels, cv2.COLOR_GRAY2BGR) * 10
        # labels = cv2.applyColorMap(heat_thold, cv2.COLORMAP_HSV)
        src = pip(src, labels, position, size, 5,
                  "Labels")
        return src


if __name__ == "__main__":
    #import cv2
    import matplotlib.pyplot as plt
    from moviepy.editor import VideoFileClip

    if False:
        test_img = cv2.imread("./test_images/scene/test5.jpg")
        carfinder = CarFinder()
        carfinder.find(test_img)
        bboxes = carfinder.draw_bounding_boxes(test_img)
        cv2.imshow('video', bboxes)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    # Testing with video clip
    if True:
        #clip = VideoFileClip('../project_video.mp4')
        clip = VideoFileClip('../test_video.mp4')
        clip_iterator = clip.iter_frames()

        carfinder = CarFinder()
        for frame in clip_iterator:
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            carfinder.find(image)
            bboxes = carfinder.draw_bounding_boxes(image)

            # Draw per frame heatmap
            bboxes = carfinder.pip_heatmap_per_frame(bboxes, (10, 5), (213, 120))

            # draw averaged heatmap
            bboxes = carfinder.pip_heatmap_averaged(bboxes, (10, 150), (213, 120))

            # Draw heatmap threshold
            bboxes = carfinder.pip_heatmap_threshold(bboxes, (250, 5), (213, 120))

            # Draw labels
            bboxes = carfinder.pip_labels(bboxes, (250, 150), (213, 120))


            cv2.imshow('video', bboxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
