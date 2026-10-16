import cv2
import numpy as np
from PIL import Image

DPI = 300
A4 = (2480, 3508)                       # A4 at 300 dpi, width x height in pixels
mm = lambda x: int(round(x * DPI / 25.4))

def save_centered(img, name):
    page = np.full((A4[1], A4[0]), 255, np.uint8)
    h, w = img.shape
    y, x = (A4[1] - h) // 2, (A4[0] - w) // 2
    page[y:y + h, x:x + w] = img
    Image.fromarray(page).save(name, resolution=DPI)

# 1) table tag: 100 mm, dictionary 4x4, id 0
tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
tag = cv2.aruco.generateImageMarker(tag_dict, 0, mm(100))
save_centered(tag, "tag_100mm.pdf")

# 2) calibration board: 5 x 7 squares of 35 mm, markers of 26 mm
board_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((5, 7), 0.035, 0.026, board_dict)
img = board.generateImage((mm(35) * 5, mm(35) * 7), marginSize=0, borderBits=1)
save_centered(img, "charuco_5x7_35mm.pdf")

print("wrote tag_100mm.pdf and charuco_5x7_35mm.pdf")