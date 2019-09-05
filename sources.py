import cv2
import random
from cropper import Cropper
from mss import mss
import numpy as np
import pafy
from vidgear.gears import CamGear
from youtube_dl import YoutubeDL


def screen_capture(cropper):
    capture_params = cropper.capture_params
    sct = mss()
    while True:
        img_raw = sct.grab(capture_params)
        img = np.asarray(img_raw)
        yield img


def video_capture(fn, cropper):
    cap = cv2.VideoCapture(fn)
    # Read until video is completed
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        yield cropper.crop(img)
    cap.release()


def video_capture_fast(fn, cropper):
    stream = CamGear(source=fn)
    stream.stream.set(cv2.CAP_PROP_POS_FRAMES, 5)
    stream.start()
    while True:
        img = stream.read()
        if img is None:
            break
        yield cropper.crop(img)
    stream.stop()


def stream_capture(url, cropper):
    if 'twitch' in url:
        url = YoutubeDL().extract_info(url, download=False)['url']
    elif 'youtube' in url:
        url = get_youtube_url(url)
    stream = CamGear(source=url).start()
    while True:
        img = stream.read()
        if img is None:
            break
        yield cropper.crop(img)
    stream.stop()


def get_cropper_video(fn):
    cap = cv2.VideoCapture(fn)
    # Read until video is completed
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # try to get the board in random frames
    all_frames = range(0, total)
    random.shuffle(all_frames)
    for frame in all_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Display the resulting frame
        try:
            cropper = Cropper(gray)
        except:
            continue
        break
    cap.release()
    return cropper


def get_cropper_capture():
    import time
    while True:
        try:
            sct = mss()
            img = sct.grab(sct.monitors[0])
            img = np.asarray(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cropper = Cropper(gray)
        except:
            time.sleep(1)
            continue
        break
    return cropper


def get_youtube_url(url):
    # copied from camgear but with a pafy bugfix thats not live on master
    source_object = pafy.new(url)
    _source = source_object.getbestvideo("mp4", ftypestrict=False)
    source = _source.url
    if source.startswith("https://manifest.googlevideo.com"):
        source = _source._info['fragment_base_url']
    return source


def get_cropper_stream(url):
    if 'twitch' in url:
        url = YoutubeDL().extract_info(url, download=False)['url']
    elif 'youtube' in url:
        url = get_youtube_url(url)
    stream = CamGear(source=url).start()  # YouTube Video URL as input
    cropper = None
    while True:
        img = stream.read()
        if img is None:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            cropper = Cropper(gray)
        except:
            continue
        break
    return cropper
