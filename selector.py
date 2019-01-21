import pygame
from pygame.locals import *
from read_digits import *
from regions import *
import win32api
import win32con
import win32gui
import pygameWindowInfo
import os

# TODO hook into the main capture.py

def selector():
    os.environ['SDL_VIDEO_WINDOW_POS'] = '8,31'
    pygame.init()
    screen = pygame.display.set_mode((1000, 750), RESIZABLE)
    winInfo = pygameWindowInfo.PygameWindowInfo()
    # del os.environ['SDL_VIDEO_WINDOW_POS']
    done = False
    fuchsia = (255, 0, 128)  # Transparency color

    # Set window transparency color
    hwnd = pygame.display.get_wm_info()["window"]
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                           win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*fuchsia), 0, win32con.LWA_COLORKEY)

    myimage = pygame.image.load("template2.png")
    screen.fill((0, 0, 0))
    screen.blit(myimage, (0, 0))
    w = 1000
    h = 750

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if event.type == QUIT:
            ret = winInfo.getScreenPosition()
            ret.update({'width': w, 'height': h})
            del ret['right']
            del ret['bottom']
            pygame.display.quit()
            return ret
        elif event.type == VIDEORESIZE:
            screen = pygame.display.set_mode(event.dict['size'], RESIZABLE)
            screen.blit(pygame.transform.scale(myimage, event.dict['size']), (0, 0))
            w = event.dict["w"]
            h = event.dict["h"]
            pygame.display.flip()
        winInfo.update()
        pygame.display.update()


if __name__ == '__main__':
    x = selector()
    print x
