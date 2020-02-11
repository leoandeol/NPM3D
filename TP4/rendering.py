import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)
normal_np = normal_np[:,:,:3]
normal_np = normal_np/255
print(np.max(normal_np))

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
        brdf = self.specular * ((flatImage @ wh.T) ** self.shininess)
        return brdf
        
    def get_cook_torrance(self, normal, wi, wo, alpha, F0, roughness):
        wh = (wi+wo)/np.linalg.norm(wi+wo)
        wh = wh[None,:]
        denom = 4 * (normal @ wi.T ) * ( normal @ wo.T )
        D = alpha**2 / (np.pi*((normal @ wh.T)**2)*(alpha**2-1)+1)
        F = F0 + (1-F0)*np.pow(2,(-5.55473*(wi@wh.T)-6.98316)*(wi @ wh))
        k = (roughness+1)**2 / 8 
        G1 = lambda v: (normal@v.T)/((normal@v.T)*(1-k)+k)
        G = G1(wi) * G1(wo)
        return (D * F * G)/denom
        

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
        else :
            brdf = material.get_lambert() + material.get_blinn_phong(normals, source.coord, wo)
        li = source.get_li()[None,:]
        image += (li * brdf * (flatImage @ source.coord[:,None])).clip(0,1)
    #image /= len(lightSources)
    image = image.reshape(normals.shape)
    return image # skrattar du forlorar du
        
if __name__ == "__main__":
    albedo = np.array([.75,0.9,0.6])
    kd = 1
    spec = 1
    shin = 1
    
    mat = Material(albedo,kd,spec,shin)
    sources = [
        LightSource(np.array([1,0.5,-1]),np.array([1.,0.,0.]),1.),
        LightSource(np.array([-1,3,1]),np.array([0.,0.,1.]),1.)
    ]

    image = shade(normal_np, mat, "lambert", sources, [0,0,0])
    plt.imshow(image)
    plt.show()
