import numpy as np
import pygame
import cv2
import json
import random
import time
import pafy
from mss import mss
from read_digits import *
from regions import *
from cropper import *
from vidgear.gears import CamGear
from youtube_dl import YoutubeDL
import os


def disp_tetris(field, screen, w, h):
    y_block_size = h / 20
    x_block_size = w / 10
    # assuming a 10 by 20 field
    for y in range(0, 20):
        for x in range(0, 10):
            if field[y][x]:
                top_left_y = y * y_block_size
                top_left_x = x * x_block_size
                block_rect = pygame.Rect(top_left_x, top_left_y, x_block_size, y_block_size)
                pygame.draw.rect(screen, (0, 255, 0), block_rect)
                pygame.draw.rect(screen, (255, 255, 255), block_rect, 1)
    return True


def img_to_tetris_array(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    h, w, _ = img.shape

    # figure out the pixel size of each block on the tetris grid, and loop over it
    # just relying on integer division has some rounding issues (always goes to floor)
    # so divide floats and round to nearest int
    y_block_size = float(h) / 20
    x_block_size = float(w) / 10
    field = np.zeros((20, 10))
    for y in range(0, 20):
        for x in range(0, 10):
            y_pos = int(round(y * y_block_size + (y_block_size / 2)))
            x_pos = int(round(x * x_block_size + (x_block_size / 2)))
            if thresh[y_pos, x_pos]:
                field[y][x] = 1
    return field


def is_clean_board(board):
    """
    returns if the board is clean or not
    it only checks for pieces in the rows past the first two
    nes tetris pieces all start in the first two rows
    """
    return not board[2:].any()


def check_end_board(board):
    """
    returns if the board indicates the game is over
    a cheap check is the top 3 rows being full - this will indicate the end "shutter" animation
    """
    return board[0:3].all()


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
    stream = CamGear(source=fn).start()
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


def init_pygame(capture_params):
    os.environ['SDL_VIDEO_WINDOW_POS'] = '%s,%s' % (
        capture_params['left'] + capture_params['width'] + 5,
        max(capture_params['top'], 40)
    )
    pygame.init()
    pygame.display.set_caption("NESTetrisCapture")
    screen = pygame.display.set_mode((375, 650))
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()
    return screen, font, clock


def capture_tetris(source, capture_params, display=True, fps_limit=60):
    done = False
    screen = font = clock = None
    get_template_digits()
    set_scales(capture_params['width'], capture_params['height'])
    if display:
        screen, font, clock = init_pygame(capture_params)
    is_playing = False
    current_game = {}
    games = []
    for img in source:
        if done:
            break
        lines = get_lines(img)
        lines_string = extract_digits(lines, "lines", template=False, length=3, letters=False)
        board = get_board(img)
        field = img_to_tetris_array(board)
        score = get_score(img)
        score_string = extract_digits(score, "score", template=False, length=6)
        if lines_string and int(lines_string) == 0 and not is_playing and is_clean_board(field):
            print "game started"
            is_playing = True
            current_game = {
                'lines': 0,
                'score': 0,
                'line_history': [0],
                'score_history': [(0, 0)],
                'level': -1
            }
        # main logic loop:
        if is_playing:
            score = get_score(img)
            score_string = extract_digits(score, "score", template=False, length=6)
            lines_num = int(lines_string)
            score_num = int(score_string)
            if lines_num > 500 or check_end_board(field):
                is_playing = False
                games.append(current_game)
                print "game ended: %s (%s)" % (current_game['score'], current_game['lines'])
                continue
            if current_game['score'] != score_num:
                current_game['score'] = score_num
                current_game['score_history'].append((lines_num, score_num))
            if current_game['lines'] != lines_num:
                current_game['lines'] = lines_num
                current_game['line_history'].append(lines_num)
        if display:
            h, w, _ = board.shape
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        cv2.imwrite("board_debug.png", board)
                        cv2.imwrite("lines_debug.png", lines)
                        cv2.imwrite("score_debug.png", score)
                    if event.key == pygame.K_p:
                        print json.dumps(current_game, indent=4)
            screen.fill((0, 0, 0))
            if fps_limit:
                clock.tick(fps_limit)
            if not disp_tetris(field, screen, w, h):
                break
            score_text = font.render(score_string, True, pygame.Color('white'))
            lines_text = font.render(lines_string, True, pygame.Color('white'))
            if fps_limit:
                fps = font.render(str(int(clock.get_fps())), True, pygame.Color('white'))
                screen.blit(fps, (0, 0))
            screen.blit(score_text, (200, 0))
            screen.blit(lines_text, (200, 35))
            pygame.display.flip()
    return games


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


def run_video(fn):
    cropper = get_cropper_video(fn)
    source = video_capture_fast(fn, cropper)
    start = time.time()
    games = capture_tetris(source, cropper.capture_params, display=False, fps_limit=None)
    print "took %s seconds" % (time.time() - start)
    return games


def run_capture():
    cropper = get_cropper_capture()
    capture_params = cropper.capture_params
    source = screen_capture(cropper)
    games = capture_tetris(source, capture_params)
    return games


def run_stream(url):
    cropper = get_cropper_stream(url)
    capture_params = cropper.capture_params
    source = stream_capture(url, cropper)
    start = time.time()
    games = capture_tetris(source, capture_params, display=False, fps_limit=None)  #, display=True, fps_limit=None)
    print "took %s seconds" % (time.time() - start)
    return games


if __name__ == '__main__':
    # games = run_stream('https://www.youtube.com/watch?v=07lPJF4-pRc')
    games = run_stream('https://www.twitch.tv/videos/444619406')
    for game in games:
        print json.dumps(game)