import tensorflow as tf
import data_collection as dc
import time

INPUT_HEIGHT = 120
INPUT_WIDTH = 10
INPUT_DEPTH = 3

KERNEL_HEIGHT = 5
KERNEL_WIDTH = 5
KERNEL_1_IN_CHANNEL = 3
KERNEL_1_OUT_CHANNEL = 32
KERNEL_2_OUT_CHANNEL = 64

FULLY_CONNECTED_1_OUTPUTS = 1024
FULLY_CONNECTED_2_OUTPUTS = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 1, 1], padding='SAME')


if __name__ == '__main__':

    # Timer
    start_time = time.time()

    # Placeholder
    x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # First layer - convolution
    W_conv1 = weight_variable([KERNEL_HEIGHT, KERNEL_WIDTH, KERNEL_1_IN_CHANNEL, KERNEL_1_OUT_CHANNEL])
    b_conv1 = bias_variable([KERNEL_1_OUT_CHANNEL])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # Second layer - 2x2 pooling
    h_pool1 = max_pool_2x2(h_conv1)

    # Third layer - convolution
    W_conv2 = weight_variable([KERNEL_HEIGHT, KERNEL_WIDTH, KERNEL_1_OUT_CHANNEL, KERNEL_2_OUT_CHANNEL])
    b_conv2 = bias_variable([KERNEL_2_OUT_CHANNEL])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Fourth layer - 2x1 pooling
    h_pool2 = max_pool_2x1(h_conv2)

    # Fifth layer - fully connected layer (30*5*64) -> (1024)
    W_fc1 = weight_variable([30 * 5 * KERNEL_2_OUT_CHANNEL, FULLY_CONNECTED_1_OUTPUTS])
    b_fc1 = bias_variable([FULLY_CONNECTED_1_OUTPUTS])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 30 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Sixth layer - fully connected layer (1024) -> (1)
    W_fc2 = weight_variable([FULLY_CONNECTED_1_OUTPUTS, FULLY_CONNECTED_2_OUTPUTS])
    b_fc2 = bias_variable([FULLY_CONNECTED_2_OUTPUTS])

    # Training
    y_conv = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y_conv), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    # init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = dc.get_data(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d, training accuracy %.3f" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xs, batch_ys = dc.get_data(500)
    print("test accuracy %.3f" % accuracy.eval(session=sess,
                                               feed_dict={x: batch_xs, y_: batch_ys}))
    print("Working time")
    print("--- %s seconds ---" % (time.time() - start_time))
