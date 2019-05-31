import cv2
import numpy as np


class GameScene:

    def __init__(self, width, height, name):
        self.name = name
        self.width = width
        self.height = height
        self.scene = np.zeros((width, height, 3), np.uint8)

    def clear_scene(self):
        self.scene = np.zeros((self.width, self.height, 3), np.uint8)

    def write_circle(self, x, y, r):
        cv2.circle(self.scene, (x, y), r, (77, 255, 9), 2, 8)

    def show_scene(self):
        cv2.imshow(self.name, cv2.cvtColor(self.scene, cv2.COLOR_RGB2BGR))

