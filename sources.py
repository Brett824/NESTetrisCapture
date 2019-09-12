import cv2
import random
from cropper import Cropper
from mss import mss
import numpy as np
import pafy
from vidgear.gears import CamGear
from youtube_dl import YoutubeDL
from threading import Thread
import Queue
import time


class VideoBuffer(object):
    def __init__(self, source, delay=0, buffer_size=1500, most_recent_frame=False):
        self.source = source
        self.delay = delay
        self.buffer_size = buffer_size
        self.thread = None
        self.is_stopping = False
        self.frame = None
        self.most_recent_frame = most_recent_frame
        self.reset()

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.is_stopping:
                break

            ret, frame = self.stream.read()

            if not ret:
                self.is_stopping = True
                self.queue.put(None, block=True, timeout=10)
                self.frame = None
                continue

            if not self.most_recent_frame or self.queue.qsize() == 0:
                self.queue.put(frame)
            else:
                self.frame = frame

        self.stream.release()

    def iter_frames(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def read(self):
        if self.most_recent_frame and self.frame is not None:
            return self.frame
        else:
            for i in range(0, 2):
                try:
                    return self.queue.get(block=True, timeout=10)
                except Queue.Empty:
                    continue
            return None

    def reset(self):
        self.queue = Queue.Queue(maxsize=self.buffer_size)
        self.stream = cv2.VideoCapture(self.source)
        if self.delay:
            time.sleep(self.delay)
        self.thread = None
        self.is_stopping = False

    def stop(self):
        self.is_stopping = True
        self.queue.put(None, block=False)
        if self.thread is not None:
            self.thread.join()


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
    stream = VideoBuffer(source=fn, buffer_size=1500)
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
    stream = VideoBuffer(source=url).start()
    for img in stream.iter_frames():
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
    stream = VideoBuffer(source=url, most_recent_frame=True).start()  # YouTube Video URL as input
    cropper = None
    for img in stream.iter_frames():
        if img is None:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            cropper = Cropper(gray)
        except Exception:
            continue
        break
    stream.stop()
    return cropper


if __name__ == '__main__':
    # url = "https://www.youtube.com/watch?v=YQ6KQ20Q8FI"
    # url = get_youtube_url(url)
    stream = VideoBuffer(source=url).start()