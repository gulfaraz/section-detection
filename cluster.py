import numpy as np
import tensorflow as tf

def cluster(k = 1000, featuresFile = "features.log"):
    X = np.genfromtxt(featuresFile, delimiter=",")

    start_pos = tf.Variable(X[np.random.randint(X.shape[0], size=k),:], dtype=tf.float32)
    centroids = tf.Variable(start_pos.initialized_value(), "S", dtype=tf.float32)

    points           = tf.Variable(X, 'X', dtype=tf.float32)
    ones_like        = tf.ones((points.get_shape()[0], 1))
    prev_assignments = tf.Variable(tf.zeros((points.get_shape()[0], ), dtype=tf.int64))

    p1 = tf.matmul(
        tf.expand_dims(tf.reduce_sum(tf.square(points), 1), 1),
        tf.ones(shape=(1, k))
    )
    p2 = tf.transpose(tf.matmul(
        tf.reshape(tf.reduce_sum(tf.square(centroids), 1), shape=[-1, 1]),
        ones_like,
        transpose_b=True
    ))
    distance = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(points, centroids, transpose_b=True))

    point_to_centroid_assignment = tf.argmin(distance, axis=1)

    total = tf.unsorted_segment_sum(points, point_to_centroid_assignment, k)
    count = tf.unsorted_segment_sum(ones_like, point_to_centroid_assignment, k)
    means = total / count

    is_continue = tf.reduce_any(tf.not_equal(point_to_centroid_assignment, prev_assignments))

    with tf.control_dependencies([is_continue]):
        loop = tf.group(centroids.assign(means), prev_assignments.assign(point_to_centroid_assignment))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    has_changed, cnt = True, 0
    while has_changed and cnt < 300:
        cnt += 1
        has_changed, _ = sess.run([is_continue, loop])

    res = sess.run(point_to_centroid_assignment)

    return res
