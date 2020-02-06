import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)


print(normal_np.shape)

class Material(object):

    def __init__(self):
        self.albedo = 1
        self.kd = 0.7
        self.fd = (self.albedo+self.kd)/np.pi
        
        
class LightSource(object):
    
    def __init__(self,x,y,z,r,g,b,it):
        self.coord = np.array([x,y,z])
        self.rgb = np.array([r,g,b])
        self.int = it
        
def shade(normalImage, materials, lightSources):
    for lightSource in lightSources:
        for material in materials:
            
        
if __name__ == "__main__":
    mat = Material()
    ls= LightSource(0,1,1,0,255,0,1)