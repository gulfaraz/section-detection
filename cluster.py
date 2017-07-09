import numpy as np
import pandas
import tensorflow as tf

def kMeans(iterations, labelledSet, columnPrefix="Cluster"):
    X = labelledSet.as_matrix()

    start_pos = tf.Variable(X[np.random.randint(X.shape[0], size=iterations),:], dtype=tf.float32)
    centroids = tf.Variable(start_pos.initialized_value(), "S", dtype=tf.float32)

    points           = tf.Variable(X, 'X', dtype=tf.float32)
    ones_like        = tf.ones((points.get_shape()[0], 1))
    prev_assignments = tf.Variable(tf.zeros((points.get_shape()[0], ), dtype=tf.int64))

    p1 = tf.matmul(
        tf.expand_dims(tf.reduce_sum(tf.square(points), 1), 1),
        tf.ones(shape=(1, iterations))
    )
    p2 = tf.transpose(tf.matmul(
        tf.reshape(tf.reduce_sum(tf.square(centroids), 1), shape=[-1, 1]),
        ones_like,
        transpose_b=True
    ))
    distance = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(points, centroids, transpose_b=True))

    point_to_centroid_assignment = tf.argmin(distance, axis=1)

    total = tf.unsorted_segment_sum(points, point_to_centroid_assignment, iterations)
    count = tf.unsorted_segment_sum(ones_like, point_to_centroid_assignment, iterations)
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

    return pandas.DataFrame(res, columns=[columnPrefix + str(iterations)])

def meanShift(n_updates=-1):
    X1 = tf.expand_dims(tf.transpose(input_X), 0)
    X2 = tf.expand_dims(input_X, 0)
    C = init_C

    sbs_C = tf.TensorArray(dtype=tf.float32, size=10000, infer_shape=False)
    sbs_C = sbs_C.write(0, init_C)

    def _mean_shift_step(C):
        C = tf.expand_dims(C, 2)
        Y = tf.reduce_sum(tf.pow((C - X1) / window_radius, 2), axis=1)
        gY = tf.exp(-Y)
        num = tf.reduce_sum(tf.expand_dims(gY, 2) * X2, axis=1)
        denom = tf.reduce_sum(gY, axis=1, keep_dims=True)
        C = num / denom
        return C

    if n_updates > 0:
        for i in range(n_updates):
            C = _mean_shift_step(C)
            sbs_C = sbs_C.write(i + 1, C)
    else:
        def _mean_shift(i, C, sbs_C, max_diff):
            new_C = _mean_shift_step(C)
            max_diff = tf.reshape(tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(new_C - C, 2), axis=1))), [])
            sbs_C = sbs_C.write(i + 1, new_C)
            return i + 1, new_C, sbs_C, max_diff

        def _cond(i, C, sbs_C, max_diff):
            return max_diff > 1e-5

        n_updates, C, sbs_C, _ = tf.while_loop(cond=_cond,
                                       body=_mean_shift,
                                       loop_vars=(tf.constant(0), C, sbs_C, tf.constant(1e10)))

        n_updates = tf.Print(n_updates, [n_updates])


    return C, sbs_C.gather(tf.range(n_updates + 1))
