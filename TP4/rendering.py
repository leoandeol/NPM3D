import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)
normal_np = normal_np[:,:,:3]

class Material(object):

    def __init__(self,albedo,kd,spec,shin,wi,wo):
        self.albedo = albedo
        self.kd = kd
        self.specular = spec
        self.shininess = shin

class LightSource(object):
    
    def __init__(self,wi,r,g,b,it):
        self.coord = wi
        self.rgb = np.array([r,g,b])
        self.int = it
        
def shade(normalImage, brdf, lightSource):
    flatImage = normalImage.reshape((-1,3))
    li = lightSource.rgb[None,:] * lightSource.int
    image = li * brdf * (flatImage @ lightSource.coord[:,None])
    image /= np.max(image)
    image = image.reshape(normalImage.shape)
    return image # skrattar du forlorar du

def lambert(kd,albedo):
    return ((kd*albedo)/np.pi)[None,:]

def blinn_phong(material,normal,wi,wo):
    wh = (wi+wo)/np.linalg.norm(wi+wo)
    wh = wh[None,:]
    flatImage = normal.reshape((-1,3))
    brdf = material.specular * (( flatImage @ wh.T ) ** material.shininess)
    return brdf
        
if __name__ == "__main__":
    albedo = np.array([.75,0.9,0.6])
    kd = 1
    spec = 1
    shin = 1
    
    x = 0
    y = 3
    z = -1
    
    r = 1.
    g = 1.
    b = 1.

    it = 1.

    wi = np.array([x,y,z])
    wo = np.array([0,0,0])    
    
    mat = Material(albedo,kd,spec,shin,wi,wo)    
    ls = LightSource(wi,r,g,b,it)

    lambert = lambert(mat.kd,mat.albedo)
    #blinn_phong = blinn_phong(mat,normal_np,wi,wo)

    brdf = lambert # * blinn_phong
    
    print(brdf.shape)
    
    image = shade(normal_np,brdf,ls)
    print(np.max(image))
    plt.imshow(image)
    plt.show()
