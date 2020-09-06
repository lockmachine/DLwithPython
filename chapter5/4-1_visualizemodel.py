import os
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('cats_and_dogs_small_1.h5')

# model.summary()

img_path = ''

plt.imshow(img_tensor[0])
plt.show()