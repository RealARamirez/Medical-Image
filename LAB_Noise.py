import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def FixImage(Image):
    Image = Image * np.array(Image>0)
    Image = Image * np.array(Image<1) + np.array(Image>1)
    return Image

class imageObj:
    def __init__(self, image):
        self.image = io.imread(image)
        return

    def WhiteNoiseAddition(self):
        image = self.image
        mask = np.random.random(size=image.shape)
        image = image/np.max(image) + mask
        image = FixImage(image)
        return image

    def GaussianNoiseAddition(self, mean, std):
        image = self.image
        mask = np.random.normal(loc=mean, scale=std, size = image.shape)
        image = image/np.max(image) + mask
        image = FixImage(image)
        return image

    def ImpulsiveNoiseAddition(self, ratio):
        image = self.image
        probnoise = np.random.binomial(1, 0.5, size=image.shape)
        saltandpepper = np.random.binomial(1, ratio/2, size=image.shape)
        saltnoise = saltandpepper * np.array(probnoise>0)
        saltnoise = np.array(1-saltnoise)
        peppernoise = saltandpepper * np.array(probnoise<1)
        peppernoise = np.array(1-peppernoise)
        image = image * saltnoise
        image = image * peppernoise + (1 - peppernoise)
        return image