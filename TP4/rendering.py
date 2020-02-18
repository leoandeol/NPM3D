import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

normal = Image.open("normal.png")
normal_np = np.array(normal)
normal_np = normal_np[:,:,:3]
normal_np = normal_np / 128
normal_np = normal_np - 1

class Material(object):

    def __init__(self,albedo,kd,spec,shin,alpha,F0):
        self.albedo = albedo
        self.kd = kd
        self.specular = spec
        self.shininess = shin
        self.alpha = alpha
        self.F0 = F0
        self.roughness = alpha**2

    def get_lambert(self):
        return ((self.kd * self.albedo) / np.pi)[None,:]

    def get_blinn_phong(self,normal,wi,wo):
        wh = (wi+wo)/np.linalg.norm(wi+wo)
        wh = wh[None,:]
        shininess_mat = (normal @ wh.T) ** self.shininess
        brdf = shininess_mat * self.specular
        return brdf
        
    def get_cook_torrance(self, normal, wi, wo):
        wh = (wi+wo)/np.linalg.norm(wi+wo)
        wh = wh[None,:]
        denom = 4 * (normal @ wi.T ) * ( normal @ wo.T )
        # SPECULAR D
        D = (self.alpha**2 / (np.pi* (normal @ wh.T)**2 * (self.alpha**2 -1) + 1)**2)
        # SPECULAR G
        k = (self.roughness +1)**2 / 8
        G1 = lambda v: (normal@v.T)/((normal@v.T)*(1-k)+k)
        G = G1(wi) * G1(wo)
        G = G[:,None]
        # SPECULAR F        
        F = self.F0 + (1-self.F0)*np.power(2, (-5.55473*(wo @ wh.T)-6.98346)*(wo @ wh.T))
        denom = denom[:,None]

        res = (D * F * G)/denom
        res[np.hstack((denom==0) for _ in range(3))]=0
        return res
        

class LightSource(object):
    
    def __init__(self,coord,rgb,it):
        self.coord = coord
        self.rgb = rgb
        self.it = it

    def get_li(self):
        return self.rgb * self.it
        
def shade(normals, material, brdf_s, lightSources, wo):
    flatNormals = normals.reshape((-1,3))
    image = np.zeros(flatNormals.shape);
    for source in lightSources:
        if brdf_s == "lambert":
            brdf = material.get_lambert()
        elif brdf_s == "blinn_phong":
            brdf = material.get_lambert() + material.get_blinn_phong(flatNormals, source.coord, wo)
        elif brdf_s == "cook":
            brdf = material.get_lambert() + material.get_cook_torrance(flatNormals, source.coord, wo)
        li = source.get_li()[None,:]
        image += ((flatNormals @ source.coord[:,None]) * brdf * li).clip(0,1)
    image = image.clip(0,1)
    image = image.reshape(normals.shape)
    return image # skrattar du forlorar du
        
if __name__ == "__main__":
    # Lambert params
    albedo = np.array([.9,0.9,0.9])
    kd = 0.8

    # Blinn_phong params
    shin = 0.1
    spec = 0.5

    # Micro facet params
    micro_facet_type = "gold"
    if micro_facet_type == "iron":
        F0 = np.array([[0.56, 0.57, 0.58]])
    elif micro_facet_type == "gold":
        F0 = np.array([[0.7, 0.4, 0.]])

    alpha = 0.7
    # Material
    mat = Material(albedo,kd,spec,shin,alpha,F0)

    # LightSources
    wi_1 = np.array([0,1,1])
    wi_1 = wi_1 / np.linalg.norm(wi_1)

    wi_2 = np.array([0,0,1])
    wi_2 = wi_2 / np.linalg.norm(wi_2)

    lightSource_color = np.array([1,1,1])
    lightSource_intensity = 1

    sources = [
        LightSource(wi_1,
                    np.array([1,0,0]),
                    lightSource_intensity),
        LightSource(wi_2,
                    np.array([1,1,1]),
                    lightSource_intensity)
    ]

    image = shade(normal_np, mat, "cook", sources, np.array([0,1,1]))
    plt.imshow(image)
    plt.show()
