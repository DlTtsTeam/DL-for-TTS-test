import PIL.ImageOps as ops
import PIL.Image as pilimage
import numpy as np
# keras api
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.models import *
from os import listdir

i_size = 28
in_size = 20
threshold = 150
picsPath = './pics/'


def load_and_process_image(path):
    img = image.load_img(path, grayscale=True, target_size=(i_size, i_size), interpolation="bicubic")
    img = ops.autocontrast(img)
    img = ops.invert(img)
    img = img.point(lambda p: 0 if p < threshold else p)
    img = crop_black_lines(image.img_to_array(img))
    img = image.array_to_img(img)
    # add re-sizing
    img = resize(img, (in_size, in_size), pilimage.BICUBIC)
    x, y = img.size
    # to square picture and add outer black lines
    img = ops.expand(img, ((i_size - x) // 2, (i_size - y) // 2))

    # img.save("./pics/test.jpg")
    return img


def crop_black_lines(img):
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    return img


def resize(img, size, resample):
    img = img.copy()

    x, y = img.size
    if x > y:
        y = int(y * size[0] / x)
        y = y if y % 2 == 0 else y + 1
        x = int(size[0])
    else:
        x = int(x * size[1] / y)
        x = x if x % 2 == 0 else x + 1
        y = int(size[1])
    size = x, y

    if size == img.size:
        return img

    img.draft(None, size)
    im = img.resize(size, resample)

    img.im = im.im
    img.mode = im.mode
    img.size = size

    img.readonly = 0
    img.pyaccess = None

    return img


# load the images
files = listdir(picsPath)
imgs = []
print(files)
for file in files:
    img = load_and_process_image(picsPath + file)
    img = image.img_to_array(img)
    imgs.append((img, file))

# using the model
model = load_model("./save/сnn5.h5")

for img in imgs:
    res = model.predict(img[0][None,:,:,:], batch_size=None, verbose=0, steps=None)
    res_n = np.argmax(res, axis=1)

    print("result: ", res, res_n, img[1])
