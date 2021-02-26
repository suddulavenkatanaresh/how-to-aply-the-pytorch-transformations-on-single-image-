import numpy
from torchvision import transforms 

from PIL  import Image 


im=Image.open('/home/vihal_venkat/Desktop/car.jpeg')




# i want to aplly the various transformations like resize,centercrop,grayscale etc on image 

transforms=transforms.Compose([  transforms.Resize(250), 
                               transforms.CenterCrop(224),
                               transforms.Grayscale(num_output_channels=3)
                               ])


im=transforms(im)



im.show()
im.save('result.jpg')
