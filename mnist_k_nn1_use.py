import tensorflow as tf
import numpy as numpy
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# keras api
from tensorflow.python.keras.models import *

print("Tensorflow version " + tf.__version__)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

print("train examples", mnist.train.num_examples)
print("test examples", mnist.test.num_examples)

i_size = 28
o_size = 10

batch = 100
epochs = 20

model = load_model("./save/nn1.h5")
res = model.predict(mnist.test.images[0:10], batch_size=None, verbose=0, steps=None)

res = numpy.argmax(res, axis=1)


print("result", res)

