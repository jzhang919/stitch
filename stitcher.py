
import numpy as np
import cv2
from matchers import Matcher
import timeit


class CVStitcher:
    def __init__(self, debug=False):
        # determine if we print debug statements and initialize the
        # cached homography matrix
        self.cachedH = None
        self.debug = debug

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        # unpack the images
        (imageB, imageA) = images

        # if the cached homography matrix is None, then we need to
        # apply keypoint matching to construct it
        if self.cachedH is None:
            # detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,
                                    featuresA, featuresB, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                return None
            if self.debug:
                print("# Matches: {}".format(len(M[0])))
                print("Homography matrix:")
                print(M[1])

            # cache the homography matrix
            self.cachedH = M[1]

        # apply a perspective transform to stitch the images together
        # using the cached homography matrix
        result = cv2.warpPerspective(imageA, self.cachedH,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None


class Stitcher:
    def __init__(
        self,
        debug,
        crop_x_min=None,
        crop_x_max=None,
        crop_y_min=None,
        crop_y_max=None,
    ):

        self.matcher_obj = Matcher()
        self.homography_cache = {}
        self.overlay_cache = {}

        self.count = 0
        self.debug = debug

        self.crop_x_min = crop_x_min
        self.crop_x_max = crop_x_max
        self.crop_y_min = crop_y_min
        self.crop_y_max = crop_y_max

    def stitch(self, images=[]):
        """
        stitches the images into a panorama
        """
        self.images = images
        self.count = len(images)
        if self.debug:
            print("# of images: {}.".format(self.count))
        self.prepare_lists()

        # left stitching
        start = timeit.default_timer()
        self.left_shift()
        self.right_shift()
        stop = timeit.default_timer()
        duration = stop - start
        print("stitching took %.2f seconds." % duration)

        if self.crop_x_min and self.crop_x_max and self.crop_y_min and self.crop_y_max:
            return self.result[
                self.crop_y_min: self.crop_y_max, self.crop_x_min: self.crop_x_max
            ]
        else:
            return self.result

    def prepare_lists(self):

        # reset lists

        self.center_index = int(self.count / 2)

        self.result = self.images[self.center_index]
        self.left_list = self.images[0:self.center_index + 1]
        self.right_list = self.images[self.center_index + 1:]

    def get_homography(self, image_1, image_1_key, image_2, image_2_key, direction):
        # TODO: use image indexes from the input array
        """
        Calculate the homography matrix between two images.
        Return from cache if possible.
        Args:
            image_1 (np.array) - first image
            image_1_key (str) - identifier for cache
            image_2 (np.array) - second image
            image_2_key (str) - identifier for cache
            direction (str) - "left" or "right"
        Returns:
            homography (np.array) - Homograpy Matrix
        """

        cache_key = "_".join([image_1_key, image_2_key, direction])
        homography = self.homography_cache.get(cache_key, None)
        if homography is None:
            # TODO: is the homography the same regardless of order??
            homography = self.matcher_obj.match(image_1, image_2)
            # put in cache
            self.homography_cache[cache_key] = homography
        return homography

    def left_shift(self):
        """
        stitch images center to left
        """
        # start off with center image
        a = self.left_list[0]

        for i, image in enumerate(self.left_list[1:]):
            H = self.get_homography(a, str(i), image, str(i + 1), "left")

            # inverse homography
            XH = np.linalg.inv(H)

            ds = np.dot(XH, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]

            f1 = np.dot(XH, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]

            XH[0][-1] += abs(f1[0])
            XH[1][-1] += abs(f1[1])

            ds = np.dot(XH, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))

            # dimension of warped image
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)

            tmp = cv2.warpPerspective(
                a, XH, dsize, borderMode=cv2.BORDER_TRANSPARENT)

            # punch the image in there
            tmp[
                offsety: image.shape[0] + offsety, offsetx: image.shape[1] + offsetx
            ] = image

            a = tmp

        self.result = tmp

    def right_shift(self):
        """
        stitch images center to right
        """
        for i, imageRight in enumerate(self.right_list):
            imageLeft = self.result

            H = self.get_homography(imageLeft, str(
                i), imageRight, str(i + 1), "right")

            # args: original_image, matrix, output shape (width, height)
            result = cv2.warpPerspective(
                imageRight,
                H,
                (imageLeft.shape[1] + imageRight.shape[1], imageLeft.shape[0]),
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            mask = np.zeros(
                (result.shape[0], result.shape[1], 3), dtype="uint8")
            mask[0: imageLeft.shape[0], 0: imageLeft.shape[1]] = imageLeft
            self.result = self.blend_images(mask, result, str(i))

    def blend_images(self, background, foreground, i):
        """
        inspired by this answer:
        https://stackoverflow.com/a/54129424/1909378
        """

        only_right = self.overlay_cache.get(i, None)
        if only_right is None:
            only_right = np.nonzero(
                (np.sum(foreground, 2) != 0) * (np.sum(background, 2) == 0)
            )
            self.overlay_cache[i] = only_right

        background[only_right] = foreground[only_right]
        return background
