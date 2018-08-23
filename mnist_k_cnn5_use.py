import PIL.ImageOps as ops
import numpy as np
# keras api
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.models import *

i_size = 28

def load_and_process_image(path):
    img = image.load_img(path, grayscale=True, target_size=(i_size, i_size), interpolation="bilinear")
    img = ops.autocontrast(img)
    img = ops.invert(img)
    img.save("test.jpg")
    return img

#load the image
img = load_and_process_image('./data/pics/77.jpg')

img = image.img_to_array(img)
#useing the model
model = load_model("./save/сnn5.h5")
res = model.predict(img[None,:,:,:], batch_size=None, verbose=0, steps=None)

res_n = np.argmax(res, axis=1)
print("result: ", res, res_n)

