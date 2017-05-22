from CarFinder.features import Features
from obsolete.lesson_functions import *
from skimage.feature import hog
from multiprocessing import Pool


class HogSubSampler:
    def __init__(self, classifier, features, scaler, ystart, ystop, scale=1, img_size=(1280, 720)):
        """
        This class implements hog subsampling and sliding window search.
        
        
        :param classifier:  Trained sklearn classifier 
        :param features: Feature object
        :param scaler: Trained sklearn feature scaler
        :param ystart: Start of vertical scanning area
        :param ystop:  Stop of vertical scanning area
        :param scale: Scaling. normal scale of search window is 64px. e.g. when 
                      scale is 2 then search window size is 128.
        :param img_size: (x, y) 
        """

        # Input image size
        self.image_size = img_size

        #self.ft = Features()
        self.ft = features
        self.clf = classifier
        self.scaler = scaler

        # Don't use feature class's hog feature vector generation,
        # it is now handled in HogSubSampler
        # self.ft.hog_feat = False

        # Y-search range
        self.ystart = ystart
        self.ystop = ystop

        # Image scaling factor
        self.scale = scale

        # Record last results here
        self.bboxes = None

    @staticmethod
    def hog_pool_wrapper(cplane_number, cplane, orientations, pixels_per_cell,
                         cells_per_block, transform_sqrt, visualise,
                         feature_vector, block_norm):
        """This function wraps a skimage hog feature extraction into form which 
        is suitable for python multiprocessing Pool workers"""
        hog_ch = hog(cplane,
                     orientations=orientations,
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block,
                     transform_sqrt=transform_sqrt,
                     visualise=visualise,
                     feature_vector=feature_vector,
                     block_norm=block_norm)
        return cplane_number, hog_ch

    def find(self, img):
        """
        https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/c3e815c7-1794-4854-8842-5d7b96276642
        
        :param img: Image of scene
        :return: 
        """
        # Crop
        img_tosearch = img[self.ystart:self.ystop, :, :]

        # Convert datatype and colorspace
        ctrans_tosearch = self.ft.convert_colorspace(img_tosearch, True)

        # Scaling is needed only when scale is different than 1
        if self.scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (
                np.int(imshape[1] / self.scale), np.int(imshape[0] / self.scale)))

        img_size = ctrans_tosearch.shape[0], ctrans_tosearch.shape[1]

        pix_per_cell= self.ft.hog_pix_per_cell[0]  # Use x for all
        cell_per_block = self.ft.hog_cell_per_block[0]  # Use x for all
        # Define blocks and steps as above
        nxblocks = (img_size[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (img_size[0] // pix_per_cell) - cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        ########
        # Generate parameters for each hog channels
        hoggs = list(self.ft.hog_channel)  # Create a placeholder for results
        params = []
        for ch in self.ft.hog_channel:
            # Wrapper has otherwise same parameters, but channel number parameter
            # is added into beginning. Params are as below.
            p = (ch, ctrans_tosearch[:, :, ch], self.ft.hog_orient_nbins,
                 self.ft.hog_pix_per_cell, self.ft.hog_cell_per_block,
                 True, False, False, 'L2-Hys')
            params.append(p)

        # Create pool of workers to extract hog features
        with Pool(processes=3) as pool:
            # Create n pool workers
            r = pool.starmap_async(HogSubSampler.hog_pool_wrapper, params)
            results = r.get(timeout=10000)
            # Put results into hoggs list in a correct order
            for result in results:
                hoggs[result[0]] = result[1]

        ###############################################
        # Sliding window search is split into 2 phases.
        #  1. Generate feature vector for each window
        #  2. Predict feature vectors as a batch
        #

        bboxes = []
        feature_vectors = []
        parameters = []

        # Generate feature vectors in this loop
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                hog_features = []
                # Extract HOG feature vector for this window
                for ch in hoggs:
                    feat = ch[ypos:ypos + nblocks_per_window,
                                      xpos:xpos + nblocks_per_window].ravel()
                    hog_features.extend(feat)

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(
                    ctrans_tosearch[ytop:ytop + window, xleft:xleft + window],
                    (64, 64))

                # Get color features
                spatial_features = self.ft.bin_spatial(subimg)
                hist_features = self.ft.color_hist(subimg)

                # Concatenate features into a 1D-feature vector
                test_features = np.concatenate((spatial_features,
                                                hist_features,
                                                hog_features)).reshape(-1)

                # Save feature vector bounding box parameters for later batch processing
                feature_vectors.append(test_features)
                parameters.append({'xbox_left': np.int(xleft * self.scale),
                                   'ytop_draw': np.int(ytop * self.scale),
                                   'win_draw': np.int(window * self.scale)})


        #with Pool(processes=4) as pool:
        # TODO: Convert below prediction code and bounding box code into a function
        #       Which can be called by pool workers.
        # Convert to numpy array for easier data conversion
        test_features = np.array(feature_vectors)
        test_features = test_features.astype(dtype=np.float64)
        # Scale and predict all feature vectors
        test_features = self.scaler.transform(test_features)
        test_prediction = self.clf.predict(test_features)

        # Create bounding box for each positive detection
        for i, pred in enumerate(test_prediction):
            if pred == 1:
                p = parameters[i]
                xbox_left = p['xbox_left']
                ytop_draw = p['ytop_draw']
                win_draw =  p['win_draw']
                pt1 = (xbox_left, ytop_draw + self.ystart)
                pt2 = (xbox_left + win_draw, ytop_draw + win_draw + self.ystart)
                bboxes.append(np.array((pt1, pt2)))

        self.bboxes = np.array(bboxes)
        hmap = self.heat_map()
        return hmap

    def draw_bounding_boxes(self, image=None, color=(0, 0, 255), thickness=3):
        """
        This method draws bounding boxes of last frame into given image or 
        if image is not given then creates a black image.
         
        :param image: Image to which bounding boxes are drawn. If empty then create black image 
        :param color: Defines color Usually (BGR), but may very depending the input image type
        :param thickness: Thickness of drawn rectangles
        :return: Image with bounding boxes.
        """
        if image is None:
            shape = (self.image_size[1], self.image_size[0], 3)
            image = np.zeros(shape, dtype=np.uint8)

        for p in self.bboxes:
            cv2.rectangle(image, tuple(p[0]), tuple(p[1]),
                          color=color, thickness=thickness)
        return image

    def heat_map(self):
        """
        Generates heatmap from the last frame results.
        
        :return: heatmap. 
        """
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
    import time
    test_img = cv2.imread("./test_images/scene/test3.jpg")

    f = Features()
    c = Classifier()

    ystart=380
    height=128
    scale=1
    ### Change parameters above
    hogss = HogSubSampler(c.classifier, f, c.scaler, ystart=ystart,
                          ystop=ystart+height, scale=scale)

    # Time finding operation
    start = time.monotonic()
    heatmap = hogss.find(test_img)
    stop = time.monotonic()
    print("hog subsampler run in {} seconds".format(stop-start))

    test_img = hogss.draw_bounding_boxes(test_img)
    #heatmap = hogss.heat_map()

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    plt.imshow(test_img)
    plt.show()
