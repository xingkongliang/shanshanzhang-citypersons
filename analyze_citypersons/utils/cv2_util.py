"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_LIGHTGREEN = (144, 238, 144)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def vis_bbox(img, bbox, color=_GREEN, thick=1):
    """Visualizes a bounding box.
    bbox mode: xywh
    """
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img


def vis_class(img, pos, class_str, color=_GRAY, font_scale=0.35):
    """Visualizes the class.
    bbox mode: xywh
    """
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    # cv2.rectangle(img, back_tl, back_br, _WHITE, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, color, lineType=cv2.LINE_AA)
    return img
