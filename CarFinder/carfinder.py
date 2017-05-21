from CarFinder.features import Features
import cv2
from CarFinder.classifier import Classifier
from CarFinder.hogsubsampler import HogSubSampler
from CarFinder.averager import Averager
from scipy.ndimage.measurements import label
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

class CarFinder:
    def __init__(self, img_size=(1280, 720)):
        self. ft = Features()
        self.clf = Classifier()
        self.image_size = img_size

        self.hogss_scale_1 = HogSubSampler(classifier=self.clf.classifier,
                                           features=self.ft,
                                           scaler=self.clf.scaler,
                                           ystart=390, ystop=518,
                                           scale=1,
                                           img_size=img_size)

        self.hogss_scale_2 = HogSubSampler(classifier=self.clf.classifier,
                                           features=self.ft,
                                           scaler=self.clf.scaler,
                                           ystart=400, ystop=528,
                                           scale=1,
                                           img_size=img_size)
        self.hogss_scale_3 = HogSubSampler(classifier=self.clf.classifier,
                                           features=self.ft,
                                           scaler=self.clf.scaler,
                                           ystart=370, ystop=626,
                                           scale=2,
                                           img_size=img_size)
        self.hogss_scale_4 = HogSubSampler(classifier=self.clf.classifier,
                                           features=self.ft,
                                           scaler=self.clf.scaler,
                                           ystart=380, ystop=636,
                                           scale=2,
                                           img_size=img_size)

        self.hoggs = [self.hogss_scale_1,
                      self.hogss_scale_2,
                      self.hogss_scale_3,
                      self.hogss_scale_4]

        # Heat map thresholding value
        self.threshold = 5
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
        heatmap  = Averager(10, np.zeros((self.image_size[1], self.image_size[0]),dtype=np.float32), True)
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

        heatmap_temp = np.zeros((self.image_size[1], self.image_size[0]))
        bound_boxes = np.zeros((self.image_size[1], self.image_size[0], 3))
        for hog in self.hoggs:
            bboxes = hog.find(image)
            #print(len(bboxes))
            hog.draw_bounding_boxes(bound_boxes)
            #cv2.imshow('bboxes', bound_boxes)
            heatmap_temp += hog.heat_map()

        # # Currently this will crash with video
        # # Works with single image
        # args = [[self, idx, image] for idx in range(len(self.hoggs))]
        # f = CarFinder.pool_wrapper_hog_find
        # with Pool(processes=4) as pool:
        #     #pool.map(f, args)
        #     results = pool.starmap(f, args)
        #     print('Waiting for tasks to complete')
        #     #pool.close()
        #     #pool.join()
        #     print('Joined')
        #     #results.wait(timeout=5000)
        #     print("collecting results")
        #     #self.hoggs = results.get(timeout=5000)
        #     print("all received")
        #     results.get(timeout=5000)
        #     print('done')

        #for hog in self.hoggs:
        #    # sum results from all hog subsamplers
        #    heatmap_temp += hog.heat_map()


        # Store per frame results, do "weak" filtering. .i.e. remove single detections
        self.heatmap_raw = heatmap_temp  # [heatmap_temp <= 1] = 0
        # Average heatmap
        self.heatmap_averaged.put(heatmap_temp.copy().astype(dtype=np.float32))
        # Threshold by setting all pixels below threshold limit to zero.
        self.heatmap_threshold = self.heatmap_raw.copy()
        self.heatmap_threshold[self.heatmap_averaged.mean() <= self.threshold] = 0
        # Generate labels and then bounding boxes
        self.labels = label(self.heatmap_threshold)
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
            pt1 = tuple(p[0])
            pt2 = tuple(p[1])
            #print(pt1, pt2)
            cv2.rectangle(image, pt1, pt2, color=color, thickness=thickness)
        return image

if __name__ == "__main__":
    #import cv2
    import matplotlib.pyplot as plt
    from moviepy.editor import VideoFileClip


    def pip(image, subimage, pos, size, border=5, title=""):
        """
        Adds sub image into image on given position and given size.

        :param image: Image to where subimage is placed 
        :param subimage: Image to be placed into image
        :param pos: Position of top left corner of subimage in image (x, y)
        :param size: Size of the subimage (It'll be resized to this size) (x, y)
        :param border: thickness of black border around subimage
        :return: combined image
        """

        # Coordinates of subimage
        x_left = pos[0] + border  # move subimage amount of border
        x_right = x_left + size[0]
        y_top = pos[1] + border
        y_bot = y_top + size[1]

        image[y_top - border:y_bot + border,
        x_left - border:x_right + border] = 0  # Cut black hole on left top corner
        image[y_top:y_bot, x_left:x_right] = cv2.resize(subimage, size,
                                                        interpolation=cv2.INTER_CUBIC)

        if len(title) != 0:
            cv2.putText(image, title, (x_left, y_top + 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

        return image


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
            frame_heatmap = carfinder.heatmap_raw.copy().astype(dtype=np.uint8)
            frame_heatmap =cv2.cvtColor(frame_heatmap, cv2.COLOR_GRAY2BGR)*20
            bboxes = pip(bboxes, frame_heatmap, (10, 5), (213, 120), 5, "Per Frame Heatmap")

            # draw averaged heatmap
            avg_heat = carfinder.heatmap_averaged.mean()
            avg_heat =  cv2.cvtColor(avg_heat.astype(dtype=np.uint8), cv2.COLOR_GRAY2BGR)*20
            bboxes = pip(bboxes, avg_heat, (10, 150), (213, 120), 5, "Average Heatmap")

            # Draw heatmap threshold
            heat_thold = carfinder.heatmap_threshold
            heat_thold = cv2.cvtColor(heat_thold.astype(dtype=np.uint8),
                                    cv2.COLOR_GRAY2BGR) * 20
            bboxes = pip(bboxes, heat_thold, (250, 5), (213, 120), 5,
                         "Heatmap Threshold")



            cv2.imshow('video', bboxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
