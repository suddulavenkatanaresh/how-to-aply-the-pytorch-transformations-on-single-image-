import numpy
from torchvision import transforms 

from PIL  import Image 


im=Image.open('/home/vihal_venkat/Desktop/car.jpeg')


transforms=transforms.Compose([  transforms.Resize(250), 
                               transforms.CenterCrop(224),
                               transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor()])


im=transforms(im)



im.show()

