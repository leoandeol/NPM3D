import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)
normal_np = normal_np[:,:,:3]

class Material(object):

    def __init__(self):
        self.albedo = np.array([1.,1.,1.])
        self.kd = 1.
        self.fd = (self.albedo*self.kd)/np.pi
        
        
class LightSource(object):
    
    def __init__(self,x,y,z,r,g,b,it):
        self.coord = np.array([x,y,z])
        self.rgb = np.array([r,g,b])
        self.int = it
        
def shade(normalImage, material, lightSource):
    flatimage = normalImage.reshape((-1,3))
    image= material.fd[None,:] * flatImage@lightSource.coord[:,None] @ lightSource.rgb[None,:] * lightSource.int
    image /= np.max(image)
    image = image.reshape(normalImage.shape)
    return image # skrattar du forlorar du
            
        
if __name__ == "__main__":
    mat = Material()
    x = 0
    y = 1
    z = 1
    r = 0.
    g = 0.8
    b = 0.4
    it = 1.
    ls = LightSource(x,y,z,r,g,b,it)
    image = shade(normal_np,mat,ls)
    plt.imshow(image)
    plt.show()
