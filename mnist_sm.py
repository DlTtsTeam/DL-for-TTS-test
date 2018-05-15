import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.feature_column.feature_column import input_layer
from tensorflow.python.tools import inspect_checkpoint as chkp

print("Tensorflow version " + tf.__version__)


def conv_layer(input, filterX, filterY, inChannels, outChannels, xStride=1, yStride=1, name='convolutional'):
    with tf.name_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([filterX, filterY, inChannels, outChannels], stddev=0.1), name="weights")
        b = tf.Variable(tf.zeros([outChannels]), name="biases")
        # preparing data for visualization
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        # model
        res = tf.nn.relu(tf.nn.conv2d(input, w, strides=[1, xStride, yStride, 1], padding="SAME") + b)
        return res


def fc_relu_layer(input, inSize, outSize, name='fc_relu'):
    with tf.name_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([inSize, outSize], stddev=0.1), name="weights")
        b = tf.Variable(tf.zeros([outSize]), name="biases")
        # preparing data for visualization
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        # model
        res = tf.nn.relu(tf.matmul(input, w) + b)
        return res


def fc_softmax_layer(input, inSize, outSize, name='fc_softmax'):
    with tf.name_scope(name) as scope:
        w = tf.Variable(tf.truncated_normal([inSize, outSize], stddev=0.1), name="weights")
        b = tf.Variable(tf.zeros([outSize]), name="biases")
        # preparing data for visualization
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        # model
        res = tf.nn.softmax(tf.matmul(input, w) + b)
        return res


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

print("train examples", mnist.train.num_examples)
print("test examples", mnist.test.num_examples)

IS = 28
L1 = 14 * 14 * 4
L2 = 7 * 7 * 16
L3 = 200
LN = 10

# initialization
x = tf.placeholder(tf.float32, [None, IS, IS, 1], name="inputs")
# placeholder for correct answers
y = tf.placeholder(tf.float32, [None, LN], name="labels")

res0 = conv_layer(x, 5, 5, 1, 4, name="layer0")
res1 = conv_layer(res0, 4, 4, 4, 8, xStride=2, yStride=2, name="layer1")
res2 = conv_layer(res1, 4, 4, 8, 16, xStride=2, yStride=2, name="layer2")
res3 = fc_relu_layer(tf.reshape(res2, [-1, L2]), L2, L3, name='layer4')
res = fc_softmax_layer(res3, L3, LN)

init = tf.global_variables_initializer()

# Success metrics:
# loss function
with tf.name_scope('ceCalc') as scope:
    crossEntropy = -tf.reduce_sum(y * tf.log(res), name="crossEntropy")
    tf.summary.scalar("crossEntropy", crossEntropy)

# % of correct answers found in batch
with tf.name_scope('aCalc') as scope:
    isCorrect = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32), name="accuracy")
    tf.summary.scalar("accuracy", accuracy)

# training step
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.003)
    trainStep = optimizer.minimize(crossEntropy)

summary = tf.summary.merge_all()

BASE_SUMMARY_DIR = "summary"
VAR_SAVER_PATH = "model/saver/vars.ckpt"
batch = 100
epoch = mnist.train.num_examples // batch
trainSteps = epoch * 8
showStep = 100
lastStep = trainSteps - 1

with tf.Session() as sess:
    sess.run(init)

    trainWriter = tf.summary.FileWriter(BASE_SUMMARY_DIR + "/train", sess.graph)
    testWriter = tf.summary.FileWriter(BASE_SUMMARY_DIR + "/test", sess.graph)

    start = time.time()
    print("start:", time.ctime(start))
    for i in range(trainSteps):
        # load batch of images and correct answers
        batchX, batchY = mnist.train.next_batch(batch)
        trainData = {x: batchX, y: batchY}

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
        testData = {x: mnist.test.images, y: mnist.test.labels}
        if i % showStep == 0 or i == lastStep:
            a, c, s = sess.run([accuracy, crossEntropy, summary], feed_dict=testData)
            testWriter.add_summary(s, i)
            print("test:", "step=", i, "accuracy=", a, "crossEntropy=", c, 'time', time.ctime(time.time()))

    print("training is done")
    print("start:", time.ctime(start))
    print("stop:", time.ctime(time.time()))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()  # use map {"W": W} to store only separate variable; we can use many savers
    saver.save(sess, VAR_SAVER_PATH)

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file(VAR_SAVER_PATH, tensor_name='', all_tensors=True, all_tensor_names=True)

with tf.Session() as sess:
    sess.run(init)

    print("\nRestoring variables...")
    saver = tf.train.Saver()  # use map {"W": W} to store only separate variable; we can use many savers
    saver.restore(sess, VAR_SAVER_PATH)

    print("Done.", "\n")
    # success on test data ?
    testData = {x: mnist.test.images, y: mnist.test.labels}
    a, c = sess.run([accuracy, crossEntropy], feed_dict=testData)
    print("Test:", "accuracy=", a, "crossEntropy=", c)
