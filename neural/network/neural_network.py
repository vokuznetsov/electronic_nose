import tensorflow as tf
import data_collection as dc


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

    x = tf.placeholder(tf.float32, [None, 120, 10, 3])

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 120, 10, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x1(h_conv2)

    W_fc1 = weight_variable([30 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 30 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_ = tf.placeholder(tf.float32, [None, 1])

    # squared_deltas = tf.square(y_conv - y_)
    # loss = tf.reduce_sum(squared_deltas)
    # optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    # gvs = optimizer.compute_gradients(loss)
    # train_step = optimizer.apply_gradients(gvs)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=y_, logits=y_conv))
    optimizer = tf.train.GradientDescentOptimizer(1e-8)
    gvs = optimizer.compute_gradients(cross_entropy)
    train_step = optimizer.apply_gradients(gvs)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y_conv), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    # init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(200):
        batch_xs, batch_ys = dc.get_train_data(), dc.get_train_labels()
        if i % 5 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d, training accuracy %.3f" % (i, train_accuracy))
            print("Y_conv_train is " + str(sess.run(tf.matmul(h_fc1, W_fc2) + b_fc2, feed_dict={x: batch_xs, y_: batch_ys})))

            test_accuracy = accuracy.eval(session=sess, feed_dict={x: dc.get_test_data(), y_: dc.get_test_labels()})
            print("step %d, training accuracy %.3f" % (i, test_accuracy))
            print("Y_conv_test is " + str(sess.run(tf.matmul(h_fc1, W_fc2) + b_fc2, feed_dict={x: dc.get_test_data(),
                                                                                               y_: dc.get_test_labels()})))

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(y_conv, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Test: train data!")
    print(sess.run(accuracy, feed_dict={x: dc.get_train_data(),
                                        y_: dc.get_train_labels()}))

    print("Test: test data!")
    print(sess.run(accuracy, feed_dict={x: dc.get_test_data(),
                                        y_: dc.get_test_labels()}))

    # print("test accuracy %g" % accuracy.eval(session=sess,
    #                                          feed_dict={x: dc.get_test_data(), y_: dc.get_test_labels(), keep_prob: 1.0}))
