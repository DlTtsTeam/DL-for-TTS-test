import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# keras api
from tensorflow.python.keras.models import *
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers

print("Tensorflow version " + tf.__version__)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

print("train examples", mnist.train.num_examples)
print("test examples", mnist.test.num_examples)

i_size = 28
o_size = 10

batch = 100
epochs = 20

inputs = Input((i_size, i_size, 1), name="input_data")
reshape = layers.Reshape((i_size * i_size,))(inputs)
outputs = layers.Dense(o_size, activation=tf.nn.softmax, name="layer1")(reshape)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizers.Adam(lr = 0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./summary/nn1', histogram_freq=1, write_grads=True, \
                                write_graph=True, write_images=True)

model.fit(x=mnist.train.images, y=mnist.train.labels, \
          batch_size=batch, epochs=epochs, validation_data=(mnist.test.images, mnist.test.labels), \
          callbacks=[cb_tensorboard])  # starts training

# test
print("test:")
result = model.evaluate(x=mnist.test.images,
                        y=mnist.test.labels)

# all model performance metrics for test
for name, value in zip(model.metrics_names, result):
    print(name, value)

model.save("./save/nn1.h5")