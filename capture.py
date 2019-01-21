import numpy as np
import pygame
import cv2
from mss import mss
from read_digits import *
from regions import *


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
    y_block_size = int(round(float(h) / 20))
    x_block_size = int(round(float(w) / 10))
    field = np.zeros((20, 10))
    for y in range(0, 20):
        for x in range(0, 10):
            y_pos = y * y_block_size + (y_block_size / 2)
            x_pos = x * x_block_size + (x_block_size / 2)
            if thresh[y_pos, x_pos]:
                field[y][x] = 1
    return field


def capture_tetris(capture_params):
    print capture_params
    sct = mss()
    done = False
    pygame.init()
    pygame.display.set_caption("test")
    get_template_digits()
    screen = pygame.display.set_mode((capture_params['width'], capture_params['height']))
    set_scales(capture_params['width'], capture_params['height'])
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill((0, 0, 0))
        clock.tick(60)
        img_raw = sct.grab(capture_params)
        img = np.asarray(img_raw)
        score = get_score(img)
        score_string = extract_digits(score, "score", template=False, length=6)
        board = get_board(img)

        field = img_to_tetris_array(board)

        h, w, _ = board.shape
        if not disp_tetris(field, screen, w, h):
            break
        fps = font.render(str(int(clock.get_fps())), True, pygame.Color('white'))
        score_text = font.render(score_string, True, pygame.Color('white'))
        screen.blit(fps, (0, 0))
        screen.blit(score_text, (200, 0))
        pygame.display.flip()


if __name__ == '__main__':
    # capture_params = get_screen_area_widget()
    capture_params = {'width': 953, 'top': 225, 'height': 662, 'left': 113}
    capture_tetris(capture_params)
    # get_template_digits()
    # img = cv2.imread("score.png")
    # get_template_digits()
    # print extract_digits(img, "score", template=False)
