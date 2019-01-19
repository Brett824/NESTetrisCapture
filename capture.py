import numpy as np
import pygame
import cv2
from mss import mss
from PIL import Image, ImageTk
from tkinter import Tk, Button, Label, Entry, LEFT, BOTTOM, RIGHT, Frame


def safe_int(val):
    if val.isdigit():
        return int(val)
    else:
        return 0


def get_screen_area_widget():
    # TODO this sucks, do something better
    root = Tk()
    # TODO reasonable defaults? monitor selection?
    capture_params = {'width': 304, 'top': 338, 'left': 464, 'height': 486}
    capture_params = {'top': 384, 'height': 540, 'width': 334, 'left': 640} # betastrep twitch
    capture_params = {'top': 378, 'height': 560, 'width': 282, 'left': 524}  # "tictactoe" twitch
    sct = mss()

    def capture_params_from_label():
        return {
            'left': safe_int(entry_x.get()),
            'top': safe_int(entry_y.get()),
            # mss yells about 0 width and height
            'width': max(safe_int(entry_width.get()), 1),
            'height': max(safe_int(entry_height.get()), 1),
        }

    def update(sct):
        new_capture = capture_params_from_label()
        if new_capture != capture_params:
            # changing this gets hugely fucked up unless you create a new mss obj
            # which requires passing it thru each loop
            capture_params.update(new_capture)
            sct = mss()
        img_raw = sct.grab(capture_params)
        img_array = np.array(img_raw)
        img_pil = Image.fromarray(img_array)
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk
        label.pack()
        root.after(100, update, sct)

    def make_entry(label_text):
        row = Frame(root)
        text = Label(row, text=label_text)
        entry = Entry(row, width=20)
        row.pack(side=BOTTOM)
        text.pack(side=LEFT)
        entry.pack(side=RIGHT)
        return entry

    label = Label(root)
    root.after(250, update, sct)
    entry_x = make_entry("X")
    entry_x.insert(0, capture_params["left"])
    entry_y = make_entry("Y")
    entry_y.insert(0, capture_params["top"])
    entry_width = make_entry("Width")
    entry_width.insert(0, capture_params["width"])
    entry_height = make_entry("Height")
    entry_height.insert(0, capture_params["height"])
    button = Button(text="Done", command=root.destroy)
    button.pack(side=BOTTOM)
    root.mainloop()
    return capture_params


def disp_tetris(field, w=100, h=200):
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


def disp_tetris2(field, screen, w, h):
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
    y_block_size = h / 20
    x_block_size = w / 10
    field = np.zeros((20, 10))
    for y in range(0, 20):
        for x in range(0, 10):
            y_pos = y * y_block_size + (y_block_size / 2)
            x_pos = x * x_block_size + (x_block_size / 2)
            if thresh[y_pos, x_pos]:
                field[y][x] = 1
    return field


def get_dominant_color(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    return palette[np.argmax(counts)]


def img_to_tetris_array(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    h, w, _ = img.shape

    # figure out the pixel size of each block on the tetris grid, and loop over it
    y_block_size = h / 20
    x_block_size = w / 10
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
    screen = pygame.display.set_mode((capture_params['width'], capture_params['height']))
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill((0, 0, 0))
        clock.tick(60)
        img_raw = sct.grab(capture_params)
        img = np.array(img_raw)
        field = img_to_tetris_array(img)
        h, w, _ = img.shape
        if not disp_tetris2(field, screen, w, h):
            break
        fps = font.render(str(int(clock.get_fps())), True, pygame.Color('white'))
        screen.blit(fps, (0, 0))
        pygame.display.flip()


if __name__ == '__main__':
    capture_params = get_screen_area_widget()
    capture_tetris(capture_params)