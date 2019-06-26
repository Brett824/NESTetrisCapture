import numpy as np
import cv2
from matplotlib import pyplot as plt


def _temp_raw_scratch(img2):
    img1 = cv2.imread('feature.png', 0)  # query Image
    # img2 = cv2.imread('%s.png'%input_fn,0)  # target Image

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    x = 444
    y = 158
    h = 644
    w = 378
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # pts = np.float32([ [x,y],[x,y+h],[x+w,y+h],[x+w,y] ]).reshape(-1,1,2)
    # pts = np.float32([ [x,y],[x+w,y],[x+w,y+h],[x,y+h] ]).reshape(-1,1,2)

    dst = cv2.perspectiveTransform(pts, M)

    # dst += (img1.shape[-1], 0)  # adding offset

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    # cv2.imshow("dsada", img3)
    # cv2.waitKey()
    return M

    # Draw bounding box in Red
    img2 = cv2.imread('%s.png' % input_fn)  # query Image
    img3 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite("result_%s.png" % input_fn, img3)


class Cropper(object):
    def __init__(self, captured_img):
        query_image = cv2.imread('feature.png', 0)

        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(query_image, None)
        kp2, des2 = orb.detectAndCompute(captured_img, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        self.transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = query_image.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, self.transform)
        self.capture_params = self.pts_to_params(dst)

    def pts_to_rect(self, pts):
        top = pts[0][0][1]
        left = pts[0][0][0]
        bottom = pts[2][0][1]
        right = pts[2][0][0]
        return top, left, bottom, right

    def pts_to_params(self, pts):
        top, left, bottom, right = self.pts_to_rect(pts)
        return {
            'top': int(top),
            'left': int(left),
            'width': int(right - left),
            'height': int(bottom - top),
        }

    def crop(self, img, pts):
        top, left, bottom, right = self.pts_to_rect(pts)
        return img[int(top):int(bottom), int(left):int(right)]

    def get_grid(self, img):
        """
        unused, poc for using this obj for region cropping
        TODO: other approach seems to work fine, so maybe port that to this class
        :param img:
        :return:
        """
        x = 444
        y = 158
        h = 644
        w = 378
        pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, self.transform)
        dst -= (self.capture_params['left'], self.capture_params['top'])
        return self.crop(img, dst)
