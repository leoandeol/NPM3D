import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)

albedo = 1
kd = 1

print(normal_np.shape)

class Material(object):

    def __init__(self):
        self.albedo = np.array([0,255,0])
        self.kd
