import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)
normal_np = normal_np[:,:,:3]
normal_np = normal_np / 128
normal_np = normal_np -1

class Material(object):

    def __init__(self,albedo,kd,spec,shin):
        self.albedo = albedo
        self.kd = kd
        self.specular = spec
        self.shininess = shin

    def get_lambert(self):
        return ((self.kd * self.albedo) / np.pi)[None,:]

    def get_blinn_phong(self,normal,wi,wo):
        wh = (wi+wo)/np.linalg.norm(wi+wo)
        wh = wh[None,:]
        flatImage = normal.reshape((-1,3))
        shininess_mat = (flatImage @ wh.T) ** self.shininess
        brdf = shininess_mat @ self.specular
        return brdf

class LightSource(object):
    
    def __init__(self,coord,rgb,it):
        self.coord = coord
        self.rgb = rgb
        self.it = it

    def get_li(self):
        return self.rgb * self.it
        
def shade(normals, material, brdf_s, lightSources, wo):
    flatImage = normals.reshape((-1,3))
    image = np.zeros(flatImage.shape);
    for source in lightSources:
        if brdf_s == "lambert":
            brdf = material.get_lambert()
        elif brdf_s == "blinn_phong":
            brdf = material.get_blinn_phong(normals, source.coord, wo)
        elif brdf_s == "all" :
            brdf = material.get_lambert() + material.get_blinn_phong(normals, source.coord, wo)
        li = source.get_li()[None,:]
        print(brdf.shape)
        image += (li * brdf * (flatImage @ source.coord[:,None]))
    #image /= len(lightSources)
    print(np.min(image))
    print(np.max(image))
    image = image.reshape(normals.shape)
    return image # skrattar du forlorar du
        
if __name__ == "__main__":
    albedo = np.array([.75,0.9,0.6])
    kd = 1
    spec = np.array([[0.8, 0.7, 0.5]])
    shin = 4
    mat = Material(albedo,kd,spec,shin)

    wi_1 = np.array([1,3,1])
    wi_1 = wi_1 / np.linalg.norm(wi_1)

    sources = [
        LightSource(wi_1,np.array([1.,1.,0.5]),1.),
    ]

    image = shade(normal_np, mat, "blinn_phong", sources, [1.,1.,1.])
    plt.imshow(image)
    plt.show()
