import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# keras api
from tensorflow.python.keras.models import *
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers

print("Tensorflow version " + tf.__version__)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

print("train examples", mnist.train.num_examples)
print("test examples", mnist.test.num_examples)

i_size = 28
l1_size = 200
l2_size = 100
l3_size = 60
l4_size = 30
o_size = 10

batch = 100
epochs = 15

inputs = Input((i_size, i_size, 1), name="input_data")
outputs = layers.Conv2D(4, kernel_size=4, strides=(1, 1), padding='same', activation=tf.nn.relu, name="layer1")(inputs)
outputs = layers.Conv2D(8, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu, name="layer2")(outputs)
outputs = layers.Conv2D(16, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu, name="layer3")(outputs)
outputs = layers.Reshape((i_size * i_size,))(outputs)
outputs = layers.Dense(l4_size, activation=tf.nn.relu, name="layer4")(outputs)
outputs = layers.Dense(o_size, activation=tf.nn.softmax, name="layer5")(outputs)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.train.AdamOptimizer(0.0003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./summary/cnn5', histogram_freq=0, \
                                write_graph=True, write_images=True)

model.fit(x=mnist.train.images, y=mnist.train.labels, \
          batch_size=batch, epochs=epochs, \
          callbacks=[cb_tensorboard])  # starts training

# test
print("test:")
result = model.evaluate(x=mnist.test.images,
                        y=mnist.test.labels)

# all model performance metrics for test
for name, value in zip(model.metrics_names, result):
    print(name, value)
