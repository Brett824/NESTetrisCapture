SCALE_X = None
SCALE_Y = None


def set_scales(w, h):
    global SCALE_X
    global SCALE_Y
    SCALE_X = w / 1000.0
    SCALE_Y = h / 750.0


# coordinates of regions in the reference template,
REGION_MAP = {
    "score": {"y": (184, 220), "x": (745, 938)},
    "board": {"y": (133, 675), "x": (372, 687)},
    "lines": {"y": (50, 80), "x": (591, 686)},
    "level": {"y": (534, 563), "x": (813, 874)},
    "tcount": {"y": (290, 325), "x": (249, 281)},
    "jcount": {"y": (344, 379), "x": (249, 281)},
    "zcount": {"y": (398, 433), "x": (249, 281)},
    "ocount": {"y": (452, 487), "x": (249, 281)},
    "scount": {"y": (506, 541), "x": (249, 281)},
    "lcount": {"y": (560, 595), "x": (249, 281)},
    "icount": {"y": (614, 649), "x": (249, 281)},
}


def get_region(img, name):
    # scale a region from the reference template to current capture
    region_x = REGION_MAP[name]["x"]
    region_y = REGION_MAP[name]["y"]
    scaled_x = tuple(int(i * SCALE_X) for i in region_x)
    scaled_y = tuple(int(i * SCALE_Y) for i in region_y)
    return img[scaled_y[0]: scaled_y[1], scaled_x[0]: scaled_x[1]]


def get_score(img):
    return get_region(img, "score")


def get_board(img):
    return get_region(img, "board")


def get_lines(img):
    return get_region(img, "lines")


def get_level(img):
    return get_region(img, "level")


def get_piece_count(img, piece):
    return get_region(img, "%scount" % piece)