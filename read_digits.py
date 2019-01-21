# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
from imutils import contours


DIGITS = {}
REGION_CACHE = {}


def get_template_digits():
    for i in range(0, 10):
        ref = cv2.imread("digits/%s.png" % i)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1]
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = imutils.grab_contours(refCnts)[0]
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = imutils.resize(roi, height=100)
        DIGITS[i] = roi
    return DIGITS


def extract_digit(img):
    scores = []
    for (digit, digitROI) in DIGITS.items():
        # apply correlation-based template matching, take the
        # score, and update the scores list
        result = cv2.matchTemplate(img, digitROI,
                                   cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        scores.append(score)
    if not score:
        return ''
    return str(np.argmax(scores))


def extract_digits(img, cachekey):
    res = ""
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1]
    ret, ref = cv2.threshold(ref, 100, 255, cv2.THRESH_BINARY)
    if not REGION_CACHE.get(cachekey):
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        if not refCnts:
            return
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
        good_rects = []
        for (i, c) in enumerate(refCnts):
            # compute the bounding box for the digit, extract it, and resize
            # it to a fixed size
            (x, y, w, h) = cv2.boundingRect(c)
            orig_roi = ref[y:y + h, x:x + w]
            if orig_roi.shape[-1] < 10:
                continue
            roi = imutils.resize(orig_roi, height=100)
            if roi.shape[-1] > 150:
                continue
            good_rects.append((x, y, w, h))
            res += extract_digit(roi)
        if res:
            REGION_CACHE[cachekey] = good_rects
        return res
    else:
        rects = REGION_CACHE.get(cachekey)
        for x, y, w, h in rects:
            orig_roi = ref[y:y + h, x:x + w]
            if orig_roi.shape[-1] < 10:
                continue
            roi = imutils.resize(orig_roi, height=100)
            if roi.shape[-1] > 150:
                continue
            res += extract_digit(roi)
        return res