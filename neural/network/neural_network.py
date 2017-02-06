import parser
import tensorflow as tf


def model():
    # Create the model
    x = tf.placeholder(tf.float32, [None, None])
    W = tf.Variable(tf.zeros([None, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # loss = tf.reduce_mean(tf.square(y_ - y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


def get_train_data():
    train = parser.get_all_standard_data()
    return train


d = get_train_data()

print 'smth'
