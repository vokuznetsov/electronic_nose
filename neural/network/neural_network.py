import tensorflow as tf
import data_collection as dc


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1],
                            strides=[1, 1, 2, 2, 1], padding='SAME')


def max_pool_2x1(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 1, 1],
                            strides=[1, 1, 2, 1, 1], padding='SAME')


if __name__ == '__main__':

    x = tf.placeholder(tf.float32, [None, 120, 10, 3])

    W_conv1 = weight_variable([3, 5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 3, 120, 10, 1])

    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x1(h_conv2)

    W_fc1 = weight_variable([30 * 5 * 3 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 30 * 5 * 3 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    y_ = tf.placeholder(tf.float32, [None, 1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(y_conv, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(10):
        batch_xs, batch_ys = dc.get_test_data(), dc.get_test_labels()
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %.3f" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    correct_prediction = tf.equal(y_conv, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Test: train data!")
    print(sess.run(accuracy, feed_dict={x: dc.get_train_data(),
                                        y_: dc.get_train_labels(), keep_prob: 1.0}))

    print("Test: test data!")
    print(sess.run(accuracy, feed_dict={x: dc.get_test_data(),
                                        y_: dc.get_test_labels(), keep_prob: 1.0}))

    # print("test accuracy %g" % accuracy.eval(session=sess,
    #                                          feed_dict={x: dc.get_test_data(), y_: dc.get_test_labels(), keep_prob: 1.0}))
