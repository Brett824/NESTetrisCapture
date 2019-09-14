import numpy as np
import imutils
import cv2
from imutils import contours


DIGITS = {}
DIGITS_BOOL = {}
REGION_CACHE = {}


def get_template_digits():
    for i in range(0, 14):
        ref = cv2.imread("digits/%s.png" % i)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1]
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = imutils.grab_contours(refCnts)[0]
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (100, 100))
        roi = imutils.resize(roi, height=100)
        DIGITS[i] = roi
        DIGITS_BOOL[i] = roi != 0
    return DIGITS


def extract_digit(img, template=True, letters=True):
    # either do correlation based template matching, take the highest scoring digit
    # or simply XOR two thresholded binary images, and take the most similar
    # template matching is significantly slower but more reliable - not good for production
    img = cv2.resize(img, (100, 100))
    scores = []
    score = 0
    if not DIGITS:
        raise Exception("Tried reading digits without initializing templates")
    diffs = {}
    img_bool = img != 0 if not template else None
    for (digit, digitROI) in DIGITS.items():
        if digit > 9 and not letters:
            continue
        if not template:
            diffs[digit] = np.count_nonzero(img_bool ^ DIGITS_BOOL[digit])
        else:
            result = cv2.matchTemplate(img, digitROI,
                                       cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
    if diffs:
        return str(min(diffs, key=diffs.get))
    if not score:
        return ''
    return str(np.argmax(scores))


def extract_digits(img, cachekey, template=True, length=None, letters=True, thresh=80):
    res = ""
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, ref = cv2.threshold(ref, thresh, 255, cv2.THRESH_BINARY)
    # use contours to find bounding boxes around each digit in the score region
    # but only do it once - the digits will always be in the same place, so just store those
    # (also always do template style matching when finding initial contours)
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
        if length is not None and len(res) != length:
            return ""
        if res:
            REGION_CACHE[cachekey] = good_rects
        return res
    else:
        rects = REGION_CACHE.get(cachekey)
        first = True
        for x, y, w, h in rects:
            orig_roi = ref[y:y + h, x:x + w]
            if orig_roi.shape[-1] < 10:
                continue
            roi = imutils.resize(orig_roi, height=100)
            if roi.shape[-1] > 150:
                continue
            res += extract_digit(roi, template=template, letters=first and letters)
            first = False
        return res


if __name__ == '__main__':
    get_template_digits()
    print extract_digits(cv2.imread("t.png"), "t", length=3)