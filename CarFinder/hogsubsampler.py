import numpy as np
import cv2
from CarFinder.features import Features
from lesson_functions import *
from skimage.feature import hog

class HogSubSampler:
    def __init__(self, classifier, features, scaler, ystart, ystop, scale=1, img_size=(1280, 720)):
        """
        
        :param classifier:  
        :param features: 
        :param scaler: 
        :param ystart: 
        :param ystop: 
        :param scale: 
        :param img_size: (x, y) 
        """

        # Defines color space conversion, if not defined then conversion
        # is not done
        self.color_space = cv2.COLOR_BGR2LUV

        # Input image size
        self.image_size = img_size

        #
        self.ft = Features()
        self.ft = features
        self.clf = classifier
        self.scaler = scaler

        # Don't use feature class's hog featere vector generation,
        # it is now handled in HogSubSampler
        # self.ft.hog_feat = False

        # Y-search range
        self.ystart = ystart
        self.ystop = ystop

        # Image scaling factor
        self.scale = scale

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        #self.window = 64

        # For hog subsampling
        #self.nblocks_per_window = (self.window // self.ft.pix_per_cell[0]) - self.ft.cell_per_block[0] + 1
        #self.cells_per_step = 2
        #nxblocks = (img_size[0] // self.ft.pix_per_cell[0]) - self.ft.cell_per_block[0] + 1
        #ysize = ystop-ystart
        #nyblocks = (ysize // self.ft.pix_per_cell[1]) - self.ft.cell_per_block[1] + 1

        #self.nxsteps = (nxblocks - self.nblocks_per_window) // self.cells_per_step
        #self.nysteps = (nyblocks - self.nblocks_per_window) // self.cells_per_step

        # Record last results here
        self.bboxes = None

    def find(self, img):
        result = self.find_cars(img,
                       ystart=self.ystart,
                       ystop=self.ystop,
                       scale=self.scale,
                       svc=self.clf,
                       X_scaler=self.scaler,
                       orient=self.ft.orient,
                       pix_per_cell=self.ft.pix_per_cell[0],
                       cell_per_block=self.ft.cell_per_block[0],
                       spatial_size=self.ft.spatial_binning_size,
                       hist_bins=self.ft.hist_nbins)
        return result

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient,
                  pix_per_cell, cell_per_block, spatial_size, hist_bins):

        #draw_img = np.copy(img)
        #img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.ft.convert_colorspace(img_tosearch)
        ctrans_tosearch = ctrans_tosearch.astype(dtype=np.float64)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (
            np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.ft.get_hog_features(ch1, feature_vec=False)
        hog2 = self.ft.get_hog_features(ch2, feature_vec=False)
        hog3 = self.ft.get_hog_features(ch3, feature_vec=False)

        bboxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window,
                            xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop + window, xleft:xleft + window],
                    (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack(
                    (spatial_features, hist_features, hog_features)).reshape(1,
                                                                             -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (
                    #xbox_left + win_draw, ytop_draw + win_draw + ystart),
                    #              (0, 0, 255), 6)
                    pt1 = (xbox_left, ytop_draw + self.ystart)
                    pt2 = (xbox_left + win_draw, ytop_draw + win_draw + self.ystart)
                    bboxes.append(np.array((pt1, pt2)))
                    #print(pt1, pt2)
        self.bboxes = np.array(bboxes)
        return bboxes

    def draw_bounding_boxes(self, image=None, color=(0, 0, 255), thickness=3):
        if image is None:
            shape = (self.image_size[1], self.image_size[0], 3)
            image = np.zeros(shape, dtype=np.uint8)

        for p in self.bboxes:
            pt1 = tuple(p[0])
            pt2 = tuple(p[1])
            print(pt1, pt2)
            cv2.rectangle(image, pt1, pt2, color=color, thickness=thickness)
        return image

    def heat_map(self):
        # https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/de41bff0-ad52-493f-8ef4-5506a279b812
        # Iterate through list of bboxes
        shape = (self.image_size[1], self.image_size[0])
        heatmap = np.zeros(shape, dtype=np.float64)
        for box in self.bboxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap


if __name__ == "__main__":
    from CarFinder.classifier import Classifier
    import matplotlib.pyplot as plt
    test_img = cv2.imread("./test_images/scene/test1.jpg")

    f = Features()
    c = Classifier()
    hogss = HogSubSampler(c.classifier, f, c.scaler, 400, 656, 1)

    bboxes = hogss.find(test_img)

    test_img = hogss.draw_bounding_boxes(test_img)
    heatmap = hogss.heat_map()
    print(heatmap.max())

    plt.imshow(heatmap)
    plt.show()
    #cv2.imshow('img', heatmap)
    #cv2.waitKey(0)
    #cv2.destroyWindow('img')