import numpy as np
from PIL import Image
i = ["a1.jpg","a1.jpeg","a1.png","a1.bmp","a1.webp"][3]
image = Image.open(i)
print(i)
print(image.format)
print(image.size)
print(image.mode)
print(np.asarray(image).shape)
print(np.asarray(image)[0][0])
print()