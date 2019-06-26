import numpy as np
import pygame
import cv2
from mss import mss
from read_digits import *
from regions import *
from cropper import *
import os


def disp_tetris_old(field, w=100, h=200):
    canvas = np.zeros((h, w, 3), np.uint8)
    y_block_size = h / 20
    x_block_size = w / 10
    # assuming a 10 by 20 field
    for y in range(0, 20):
        for x in range(0, 10):
            if field[y][x]:
                top_left_y = y * y_block_size
                top_left_x = x * x_block_size
                bottom_right_y = top_left_y + y_block_size
                bottom_right_x = top_left_x + x_block_size
                cv2.rectangle(canvas, (top_left_x, top_left_y),
                              (bottom_right_x, bottom_right_y),
                              (0, 255, 0), thickness=-1)
                cv2.rectangle(canvas, (top_left_x, top_left_y),
                              (bottom_right_x, bottom_right_y),
                              (255, 255, 255), thickness=1)
    cv2.imshow('test', canvas)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    return True


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


def capture_tetris(cropper):
    capture_params = cropper.capture_params
    sct = mss()
    done = False

    os.environ['SDL_VIDEO_WINDOW_POS'] = '%s,%s' % (
        capture_params['left'] + capture_params['width'],
        capture_params['top']
    )
    pygame.init()
    pygame.display.set_caption("test")
    get_template_digits()
    screen = pygame.display.set_mode((capture_params['width'], capture_params['height']))
    set_scales(capture_params['width'], capture_params['height'])
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()
    lines_stored = "0"
    tetris_counter = {}
    temp_line_names = ['single', 'double', 'triple', 'tetris']
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    cv2.imwrite("board_debug.png", board)
                    cv2.imwrite("lines_debug.png", lines)
                    cv2.imwrite("score_debug.png", score)
        screen.fill((0, 0, 0))
        clock.tick(60)
        img_raw = sct.grab(capture_params)
        img = np.asarray(img_raw)
        score = get_score(img)
        # cv2.imwrite("score2.png", score)
        score_string = extract_digits(score, "score", template=False, length=6)
        lines = get_lines(img)
        lines_string = extract_digits(lines, "lines", template=False, length=3)
        # happens on all black
        if lines_string and lines_string == '000':
            tetris_counter = {}
            lines_stored = lines
        if lines_string and lines_string != '777' and int(lines_string) < 350:
            line_diff = int(lines_string) - int(lines_stored)
            if line_diff < 0:  # score reset?
                print tetris_counter
                tetris_counter = {}
            elif 0 < line_diff < 5:
                tetris_counter[line_diff] = tetris_counter.get(line_diff, 0) + 1
                tetris_score = int(tetris_counter.get(4, 0)) * 4
                print tetris_counter
                if int(lines_string):
                    tetris_percent = float(tetris_score) / int(lines_string) * 100
                else:
                    tetris_percent = 0
                print "%s: %s (%%%s)" % (lines_string, temp_line_names[line_diff-1], tetris_percent)
            elif line_diff != 0:
                print "weird line diff: %s -> %s" % (lines_stored, lines_string)
            lines_stored = lines_string

        board = get_board(img)
        # cv2.imwrite("board2.png", board)
        field = img_to_tetris_array(board)
        h, w, _ = board.shape
        if not disp_tetris(field, screen, w, h):
            break
        fps = font.render(str(int(clock.get_fps())), True, pygame.Color('white'))
        score_text = font.render(score_string, True, pygame.Color('white'))
        lines_text = font.render(lines_string, True, pygame.Color('white'))
        screen.blit(fps, (0, 0))
        screen.blit(score_text, (200, 0))
        screen.blit(lines_text, (200, 35))
        pygame.display.flip()


if __name__ == '__main__':
    # capture_params = get_screen_area_widget()
    sct = mss()
    img = sct.grab(sct.monitors[0])
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropper = Cropper(gray)
    capture_tetris(cropper)
