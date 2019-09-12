import numpy as np
import pygame
from pygame.locals import DOUBLEBUF
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
from threading import Thread
import os
import datetime
from sources import *
import cProfile as profile


PIECES = ['t','j','z','o','s','l','i']


class TetrisSession(object):

    def __init__(self, capture_params, source, display=False, fps_limit=60):
        self.capture_params = capture_params
        self.source = source
        self.display = display
        self.fps_limit = fps_limit

    def start(self):
        """
        start the thread to read frames from the video stream
        """
        self.thread = Thread(target=self.run_game, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def timereport(self, msg):
        print "%s %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)

    def init_pygame(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '%s,%s' % (
            self.capture_params['left'] + self.capture_params['width'] + 5,
            max(self.capture_params['top'], 40)
        )
        pygame.init()
        pygame.display.set_caption("NESTetrisCapture")
        screen = pygame.display.set_mode((375, 650), DOUBLEBUF)
        screen.set_alpha(None)
        font = pygame.font.Font(None, 30)
        clock = pygame.time.Clock()
        self.screen = screen
        self.font = font
        self.clock = clock
        self.current_game = {}
        self.games = []

    def img_to_tetris_array(self, img):
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

    def is_clean_board(self, board):
        """
        returns if the board is clean or not
        it only checks for pieces in the rows past the first two
        nes tetris pieces all start in the first two rows
        """
        return not board[2:].any()

    def check_end_board(self, board):
        """
        returns if the board indicates the game is over
        a cheap check is the top 3 rows being full - this will indicate the end "shutter" animation
        """
        return board[0:3].all()

    def is_new_piece(self, board, img, old_counts):
        """
        returns if there's a fresh piece on the board
        check if theres anything in the top two rows, and if the piece counts on the left changed
        """
        in_top_row = board[0:2].any()
        if not in_top_row:
            return False, old_counts
        new_counts = self.get_piece_counts(img)
        found_diff = False
        for piece in PIECES:
            diff = new_counts[piece] - old_counts[piece]
            if diff in (1, -9):
                if found_diff:
                    print "multiple diffs?"
                    return False, new_counts
                else:
                    found_diff = True
            elif diff:
                print "weird diffs? (%s, %s)" % (piece, diff)
                return False, new_counts
        return found_diff, new_counts

    def is_tetris_ready(self, board):
        cropped_board = board[2:]  # cut off the top two rows which can contain the next falling piece
        # for the bottom 4 rows, map to "None" or the location of a single hole in the row
        holes = [np.where(row == 0)[0][0] if sum(row) == 9 else None for row in board[-4:]]
        # check if all the holes are in the same location and the whole column is empty
        return len(set(holes)) == 1 and holes[0] is not None and sum(cropped_board[:, holes[0]]) == 0

    def get_piece_counts(self, img):
        counts = {}
        for piece in PIECES:
            piece_img = get_piece_count(img, piece)
            # red color digits are tricky, try diff thresh
            for thresh in (20, 10):
                piece_count = extract_digits(piece_img, piece, template=False, length=1, letters=False, thresh=thresh)
                if piece_count:
                    break
            else:
                cv2.imwrite("failed.png", piece_img)
                print "failed %s" % piece
            counts[piece] = int(piece_count)
        return counts

    def run_game(self):
        done = False
        get_template_digits()
        set_scales(self.capture_params['width'], self.capture_params['height'])
        if self.display:
            self.init_pygame()
        is_playing = False
        drought = 0
        droughts = []
        was_tetris_ready = False
        self.current_game = {}
        self.games = []
        piece_counts = {}
        per_second = 0
        last_time = time.time()
        for img in self.source:
            per_second += 1
            if time.time() - last_time > 1:
                print "Per second: %s" % per_second
                last_time = time.time()
                per_second = 0
            if done:
                break
            lines = get_lines(img)
            lines_string = extract_digits(lines, "lines", template=False, length=3, letters=False)
            board = get_board(img)
            field = self.img_to_tetris_array(board)
            if lines_string and int(lines_string) == 0 and not is_playing and self.is_clean_board(field):
                level = get_level(img)
                level_string = int(extract_digits(level, "level", template=False, length=2))
                # validate level against nonhacked starts
                level_string = -1 if level_string > 19 else level_string
                self.timereport("game started (level %s)" % level_string)
                is_playing = True
                piece_counts = dict(zip(PIECES, [0] * len(PIECES)))  # initialize at 0 in an obnoxious way
                drought = 0
                droughts = []
                self.current_game = {
                    'lines': 0,
                    'score': 0,
                    'line_history': [0],
                    'score_history': [(0, 0)],
                    'level': level_string,
                    'ended': False
                }
            # main logic loop:
            if is_playing:
                score = get_score(img)
                score_string = extract_digits(score, "score", template=False, length=6)
                lines_num = int(lines_string)
                score_num = int(score_string)
                if lines_num > 500 or self.check_end_board(field):
                    is_playing = False
                    self.current_game['ended'] = True
                    self.games.append(self.current_game)
                    self.timereport("game ended: %s (%s)" % (self.current_game['score'], self.current_game['lines']))
                    if droughts:
                        self.timereport("avg drought: %s" % (sum(droughts)/float(len(droughts))))
                    continue
                # if we werent able to get the level first try, keep trying
                if self.current_game['level'] == -1:
                    level = get_level(img)
                    level_string = int(extract_digits(level, "level", template=False, length=2))
                    if level_string < 20:
                        self.current_game['level'] = level_string
                if self.current_game['score'] != score_num:
                    self.current_game['score'] = score_num
                    self.current_game['score_history'].append((lines_num, score_num))
                if self.current_game['lines'] != lines_num:
                    if lines_num - self.current_game['lines'] == 4 and was_tetris_ready:
                        print "Tetris, drought: %s" % drought
                        droughts.append(drought)
                        drought = 0
                    self.current_game['lines'] = lines_num
                    self.current_game['line_history'].append(lines_num)
                # make sure multiple frames of new pieces dont add up
                is_new_piece, piece_counts = self.is_new_piece(field, img, piece_counts)
                # drought counter logic
                if is_new_piece and self.is_tetris_ready(field):
                    drought += 1
                    was_tetris_ready = self.is_tetris_ready(field)
                self.current_game['drought'] = drought
            if self.display:
                if not self.draw(board, field, self.current_game):
                    break

    def disp_tetris(self, field, screen, w, h):
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

    def draw(self, board, field, current_game):
        h, w, _ = board.shape
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    pass  # TODO figure out this debug option
                    # cv2.imwrite("board_debug.png", board)
                    # cv2.imwrite("lines_debug.png", lines)
                    # cv2.imwrite("score_debug.png", score)
                if event.key == pygame.K_p:
                    print json.dumps(current_game, indent=4)
        self.screen.fill((0, 0, 0))
        if self.fps_limit:
            self.clock.tick(self.fps_limit)
        self.disp_tetris(field, self.screen, w, h)

        score_text = self.font.render(str(current_game.get('score')), True, pygame.Color('white'))
        lines_text = self.font.render(str(current_game.get('lines')), True, pygame.Color('white'))
        drought_text = self.font.render(str(current_game.get('drought')), True, pygame.Color('white'))
        if self.fps_limit:
            fps = self.font.render(str(int(self.clock.get_fps())), True, pygame.Color('white'))
            self.screen.blit(fps, (0, 0))
        self.screen.blit(score_text, (200, 0))
        self.screen.blit(lines_text, (200, 35))
        self.screen.blit(drought_text, (200, 70))
        pygame.display.flip()
        return True


def run_capture():
    cropper = get_cropper_capture()
    capture_params = cropper.capture_params
    source = screen_capture(cropper)
    capture = TetrisSession(capture_params, source, display=True)
    capture.start()
    return capture



def run_video(fn):
    cropper = get_cropper_video(fn)
    source = video_capture_fast(fn, cropper)
    capture = TetrisSession(cropper.capture_params, source, display=False, fps_limit=30)
    capture.start()
    return capture


def run_stream(url):
    cropper = get_cropper_stream(url)
    source = stream_capture(url, cropper)
    capture = TetrisSession(cropper.capture_params, source, display=True)
    capture.start()
    return capture


if __name__ == '__main__':
    session = run_capture()
    # session = run_stream("https://www.youtube.com/watch?v=IcO18WE0R1U")
    # session = run_video("koryan_wr.webm")
    session.thread.join()
    # session = run_stream('https://www.youtube.com/watch?v=07lPJF4-pRc')
