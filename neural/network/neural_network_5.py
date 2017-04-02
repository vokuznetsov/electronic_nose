import tensorflow as tf
import data.data_collection as dc
import time

INPUT_HEIGHT = 5
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


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')


def train_and_save(sess, start_pos, end_pos, is_save, place=""):
    for i in range(1000):
        batch_xs, batch_ys = dc.get_train_data(50, start_pos, end_pos)
        if i % 50 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d, training accuracy %.3f" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if is_save:
        save_model(sess, place)


def save_model(sess, place):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Save the variables to disk.
    save_path = saver.save(sess, "./resource/training_model/" + str(place) + "/model.ckpt")
    print("Model saved in file: %s" % save_path)


def restore_model(sess, place):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, "./resource/training_model/" + str(place) + "/model.ckpt")
    print("Model restored.")
    return sess


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

    # Second layer - 2x1 pooling
    h_pool1 = max_pool_1x2(h_conv1)

    # Third layer - fully connected layer (5*5*32) -> (1)
    W_fc1 = weight_variable([5 * 5 * KERNEL_1_OUT_CHANNEL, FULLY_CONNECTED_2_OUTPUTS])
    b_fc1 = bias_variable([FULLY_CONNECTED_2_OUTPUTS])
    h_pool1_flat = tf.reshape(h_pool1, [-1, 5 * 5 * 32])

    # Training
    y_conv = tf.nn.sigmoid(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y_conv), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    # init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    start_elem = 0
    end_elem = 5
    train_and_save(sess, start_elem, end_elem, True, "5_first")
    # sess = restore_model(sess, "5_first")

    values = []
    for i in range(0, 10):
        batch_xs, batch_ys = dc.get_test_data(50, start_elem, end_elem)
        # batch_xs, batch_ys = dc.get_test_non_stand_splitted_by_alc_data(10)
        # batch_xs, batch_ys = dc.get_test_data_for_all_alc(20)
        values.append(accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys}))
        print("test accuracy %.3f" % accuracy.eval(session=sess,
                                                   feed_dict={x: batch_xs, y_: batch_ys}))

    print("Average accuracy is %.3f" % (sum(values) / len(values)))
    print("Working time")
    print("--- %s seconds ---" % (time.time() - start_time))
