# how-to-aply-the-pytorch-transformations-on-single-image-



# fisr tt load the pytorch packages  and libraries
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




import numpy
from torchvision import transforms 

from PIL  import Image     # pl for image opening 


im=Image.open('/home/vihal_venkat/Desktop/car.jpeg')    #  then define the image path



# now compose thne transforms what ever you want to aplly on the image 
transforms=transforms.Compose([  transforms.Resize(250), 
                               transforms.CenterCrop(224),
                               transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor()])


im=transforms(im)    # then apply 
