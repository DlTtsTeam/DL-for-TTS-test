import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# initialization
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="inputs")
with tf.name_scope('mainLayer') as scope:
    W = tf.Variable(tf.zeros([784, 10]), name="weights")
    b = tf.Variable(tf.zeros([10]), name="biases")
    # model
    Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b, name="Y")

init = tf.global_variables_initializer()

# placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10], name="Y_")

# Success metrics:
# loss function
with tf.name_scope('ceCalc') as scope:
    crossEntropy = -tf.reduce_sum(Y_ * tf.log(Y), name="crossEntropy")

# % of correct answers found in batch
with tf.name_scope('aCalc') as scope:
    isCorrect = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32), name="accuracy")

# training step
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    trainStep = optimizer.minimize(crossEntropy)

# preparing data for visualization
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("crossEntropy", crossEntropy)
tf.summary.histogram("weights", W)
tf.summary.histogram("biases", b)
summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)

BASE_SUMMARY_DIR = "summary"

trainWriter = tf.summary.FileWriter(BASE_SUMMARY_DIR + "/train", sess.graph)
testWriter = tf.summary.FileWriter(BASE_SUMMARY_DIR + "/test", sess.graph)

trainSteps = 10000
showStep = trainSteps / 100
lastStep = trainSteps - 1

start = time.time()
for i in range(trainSteps):
    # load batch of images and correct answers
    batchX, batchY = mnist.train.next_batch(100)
    trainData = {X: batchX, Y_: batchY}

    # train and write statistics
    if i % showStep == 0 or i == lastStep:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        s, t = sess.run([summary, trainStep], feed_dict=trainData, options=run_options, run_metadata=run_metadata)
        trainWriter.add_summary(s, i)
        trainWriter.add_run_metadata(run_metadata, 'step%d' % i)
    else:
        s, t = sess.run([summary, trainStep], feed_dict=trainData)
        trainWriter.add_summary(s, i)

    # success on test data ?
    testData = {X: mnist.test.images, Y_: mnist.test.labels}
    if i % showStep == 0 or i == lastStep:
        a, c, s = sess.run([accuracy, crossEntropy, summary], feed_dict=testData)
        testWriter.add_summary(s, i)
        print("test:", "step=", i, "accuracy=", a, "crossEntropy=", c)

print("training is done")
print("start:", time.ctime(start))
print("stop:", time.ctime(time.time()))
