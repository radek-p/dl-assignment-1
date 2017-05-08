#!/usr/bin/env python
import sys

import math
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data as mnist_input


# from tensorflow.python import debug as tf_debug

class LayerBuilder(object):
    def __init__(self, parameters=None):
        self.parameters = {
            "epsilon": 1e-3,
        }
        if parameters is not None:
            self.parameters.update(parameters)

    def conv2d_with_bn(self, signal, filter_size, in_channels, out_channels, basename="conv_layer"):
        # create trainable variables for the layer
        W = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1),
            name="{}_W".format(basename)
        )
        gamma = tf.Variable(tf.constant(1.0, shape=[out_channels]), name="{}_gamma".format(basename))
        beta = tf.Variable(tf.constant(0.0, shape=[out_channels]), name="{}_beta".format(basename))

        # apply convolution filters
        signal = tf.nn.conv2d(signal, W, strides=[1, 1, 1, 1], padding="SAME")

        # normalize
        mean = tf.reduce_mean(signal, axis=[0, 1, 2])
        signal -= mean
        variance = tf.reduce_mean(tf.square(signal), axis=[0, 1, 2])
        signal /= tf.sqrt(variance + self.parameters["epsilon"])

        # scale and shift
        signal = gamma * signal + beta
        return signal, locals(), [W, gamma, beta]

    @staticmethod
    def max_pool(signal, factor, basename="max_pool"):
        signal = tf.nn.max_pool(signal, ksize=[1, factor, factor, 1], strides=[1, 2, 2, 1], padding="SAME",
                                name=basename)
        return signal, locals()

    @staticmethod
    def fully_connected(signal, fan_in, fan_out, basename="fully_connected_layer"):
        # create trainable variables for the layer
        W = tf.Variable(
            tf.truncated_normal([fan_in, fan_out], stddev=0.1),
            name="{}_W".format(basename)
        )
        b = tf.Variable(
            tf.constant(0.1, shape=[fan_out])
        )

        signal = tf.matmul(signal, W) + b
        return signal, locals(), [W, b]


class Trainer(object):
    def __init__(self, tf_session, input_data, parameters=None):
        self.input_data = input_data
        self.model = {}
        self.session = tf_session
        self.layers = LayerBuilder()

        self.parameters = {
            "input_size": 28,
            "class_num": 10,
            "training_steps": 10000,
            "dreaming_steps": 2000,
            "save_path_prefix": "./models3",
            "dream_points_num": 49,
            "dream_points_d": 7,
        }
        if parameters is not None:
            self.parameters.update(parameters)

    def create_dream(self, create_variables_only):
        dream_count = self.parameters["class_num"]
        dream_points_num = self.parameters["dream_points_num"]
        dream_points_d = self.parameters["dream_points_d"]
        line_segments_num = dream_points_num
        dream_size = self.parameters["input_size"]
        # big_dream_size = dream_size * 3
        big_dream_size = dream_size

        angles = tf.Variable(tf.zeros([dream_count, line_segments_num, 1, 1, 1]))
        opacities = tf.Variable(tf.ones([dream_count, line_segments_num, 1, 1, 1]))

        trainable_variables = [angles, opacities]

        if create_variables_only:
            dream = tf.ones([dream_count, dream_size, dream_size, 1]) * angles[0, 0, 0, 0, 0] * opacities[0, 0, 0, 0, 0]
            return dream, locals(), trainable_variables

        coord1 = np.broadcast_to(np.arange(big_dream_size, dtype=np.float32), [big_dream_size, big_dream_size])
        coord2 = coord1.T

        coords = np.stack([coord1, coord2], 2)
        coords = np.reshape(coords, [1, 1, big_dream_size, big_dream_size, 2, 1])

        center = np.array([14, 14])
        l = 6
        circle = np.array([
            [(1.5, 1.5), (6, 0)],
            [(1.5, 1.5), (0, 6)],
            [(10.5, 1.5), (6, 0)],
            [(10.5, 1.5), (12, 6)],

            [(1.5, 10.5), (6, 12)],
            [(1.5, 10.5), (0, 6)],
            [(10.5, 10.5), (6, 12)],
            [(10.5, 10.5), (12, 6)],
        ], dtype=np.float32)

        circle2 = circle.copy()
        circle2[:, :, 1] += 12.

        frame = np.array([
            [(0, 0), (l, 0)],
            [(l, 0), (2 * l, 0)],

            [(0, 0), (0, l)],
            [(0, l), (0, 2 * l)],
            [(2 * l, 0), (2 * l, l)],
            [(2 * l, l), (2 * l, 2 * l)],

            [(0, 2 * l), (l, 2 * l)],
            [(l, 2 * l), (2 * l, 2 * l)],

            [(0, 2 * l), (0, 3 * l)],
            [(0, 3 * l), (0, 4 * l)],
            [(2 * l, 2 * l), (2 * l, 3 * l)],
            [(2 * l, 3 * l), (2 * l, 4 * l)],

            [(0, 4 * l), (l, 4 * l)],
            [(l, 4 * l), (2 * l, 4 * l)],
        ], dtype=np.float32)

        diagonal = np.array([
            [(12, 0), (6, 24)],
            # [(12, 6), (0, 18)]
        ], dtype=np.float32)

        line_segments = np.concatenate((circle, circle2, frame, diagonal), 0)
        line_segments -= [6, 12]
        line_segments *= 0.8
        line_segments += center

        # p1 = dream_points[:, 0::2, :, :]
        # p1 = p1[:, :-1, :, :]
        # p2 = dream_points[:, 1::2, :, :]

        # p1 = dream_points[:, :-6, :, :]
        # p2 = dream_points[:, 1:-5, :, :]
        limit_sqr = 31
        # limit_sqrt = 7
        # dream_points_d = limit_sqrt

        # initial_points = np.zeros([line_segments_num, 2], dtype=np.float32)
        # for i in range(dream_points_num):
        #     # initial_points[i, :] = [(i % dream_points_d) * 4 + 2, (i // dream_points_d) * 4 + 2]
        #     initial_points[i, :] = [(i % dream_points_d) * 4. + 2., (i // dream_points_d) * 4. + 2.]
        # initial_points = np.tile(initial_points, [dream_count, 1, 1])
        # initial_points = initial_points.reshape([dream_count, dream_points_num, 1, 1, 2, 1])

        # p1 = dream_points[:, :-1, :, :]
        # p2 = dream_points[:, 1:, :, :]
        # initial_points = initial_points[:, 0:limit_sqr, :, :, :, :]
        angles = angles[:, 0:limit_sqr, :, :, :]
        opacities = opacities[:, 0:limit_sqr, :, :, :]
        line_segments_num = limit_sqr
        dream_points_num = limit_sqr

        p1 = line_segments[:, 0, :]
        p2 = line_segments[:, 1, :]

        p1 = np.tile(p1, [dream_count, 1, 1])
        p1 = p1.reshape([dream_count, dream_points_num, 1, 1, 2, 1])
        p2 = np.tile(p2, [dream_count, 1, 1])
        p2 = p2.reshape([dream_count, dream_points_num, 1, 1, 2, 1])
        # p2 = p1 + tf.stack([tf.sin(angles), tf.cos(angles)], 4) * 4.

        _v = coords - p1
        _s = p1 - p2
        _s = tf.tile(_s, [1, 1, big_dream_size, big_dream_size, 1, 1])
        g = tf.matmul(_v, _s, True)
        d = tf.matmul(_s, _s, True)
        rate = g / (d + 1e-6)
        projection = rate * _s
        projection = projection + p1

        z = p1 + p2 - 2 * projection
        d2 = tf.matmul(z, z, True)

        projects_on_segment = tf.sigmoid((tf.sqrt(d + 1e-6) - tf.sqrt(d2 + 1e-6)) * 2.)
        # projects_on_segment = tf.sigmoid(d - d2)

        projects_on_segment = tf.reshape(projects_on_segment,
                                         [dream_count, line_segments_num, big_dream_size, big_dream_size, 1])

        vecs = coords - projection
        dots = tf.matmul(vecs, vecs, True)
        dist = tf.sqrt(dots + 1e-6)
        is_close_to_segment = tf.reduce_max(tf.sigmoid((0.5 - dist) * 2.), [4])

        big_dream = projects_on_segment * is_close_to_segment * opacities
        big_dream = tf.clip_by_value(tf.reduce_sum(big_dream, 1), 0., 1.)
        # big_dream = tf.reduce_max(tf.sigmoid(projection - p1) * tf.sigmoid(p2 - projection), [1, 4])
        # print("big_dream.shape", big_dream.shape)
        # dream = tf.image.resize_bilinear(big_dream, [28, 28], name="dream")
        dream = big_dream

        return dream, locals(), trainable_variables

    def create_model(self, trimmed_for_faster_training):
        print("Creating the model")
        size_0 = self.parameters["input_size"]

        # Model switches
        is_dreaming = tf.placeholder(tf.bool, name="is_dreaming")
        keep_probability = tf.placeholder(tf.float32, name="keep_probability")

        # Network placeholders and gates
        x = tf.placeholder(tf.float32, [None, size_0, size_0, 1], name="x")
        y = tf.placeholder(tf.float32, [None, 10], name="y")

        dream, init, v0 = self.create_dream(trimmed_for_faster_training)

        signal = tf.cond(is_dreaming, lambda: dream, lambda: x)

        # Layers
        signal, cn1, v1 = self.layers.conv2d_with_bn(signal, 5, 1, 48, "cn1")
        signal = tf.nn.relu(signal)
        signal, mp1 = self.layers.max_pool(signal, 2, "mp1")
        signal, cn2, v2 = self.layers.conv2d_with_bn(signal, 5, 48, 64, "cn2")
        signal = tf.nn.relu(signal)
        signal, mp2 = self.layers.max_pool(signal, 2, "mp2")
        signal = tf.reshape(signal, [-1, 7 * 7 * 64])
        signal, fc1, v3 = self.layers.fully_connected(signal, 7 * 7 * 64, 1024, "fc1")
        signal = tf.nn.relu(signal)
        signal = tf.nn.dropout(signal, keep_probability)
        signal, fc2, v4 = self.layers.fully_connected(signal, 1024, 10, "fc2")
        result = tf.nn.softmax(signal)

        # Measures
        cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(result, 1e-10, 1.0)))
        # error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=signal, name="error"))
        error = cross_entropy

        is_correct = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1), name="is_correct")
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")

        contrast = tf.reduce_mean(-tf.multiply(dream, tf.subtract(dream, 1.)))
        white_ratio = tf.reduce_mean(dream)
        print(white_ratio.shape)
        black_ratio = 1. - white_ratio
        # opacities = tf.clip_by_value(v0[1], 0., 1.)
        opacities = v0[1]  # tf.clip_by_value(v0[1], 0., 1.)
        opacities_contrast = tf.reduce_mean(tf.square(-tf.multiply(opacities, tf.subtract(opacities, 1.))))

        # Training operations
        training_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, var_list=v1 + v2 + v3 + v4)
        # training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        dreaming_optimizer = tf.train.AdamOptimizer(0.05, name="dreaming_optimizer")
        training_step_2 = dreaming_optimizer.minimize(cross_entropy, var_list=v1 + v2 + v3 + v4)

        # optimize_white_balance = dreaming_optimizer.minimize(cross_entropy + white_ratio * 100, var_list=v0)
        dreaming_step_1 = dreaming_optimizer.minimize(
            cross_entropy + tf.square(white_ratio - 0.12) * 100 + opacities_contrast * 30, var_list=v0)
        dreaming_step_2 = dreaming_optimizer.minimize(cross_entropy, var_list=v0)

        saver = tf.train.Saver()
        self.model = locals()
        # self.model["dream_points"] = init["dream_points"]

    def preprocess_input(self, x):
        return np.reshape(x, [-1, self.parameters["input_size"], self.parameters["input_size"], 1])

    def train_model(self):
        print("Training the model")
        self.session.run(tf.global_variables_initializer())

        for step in range(self.parameters["training_steps"] + 1):
            x, y = self.input_data.train.next_batch(100)
            x = self.preprocess_input(x)

            self.session.run(
                fetches=self.model["training_step"],
                feed_dict={
                    self.model["x"]: x,
                    self.model["y"]: y,
                    self.model["is_dreaming"]: False,
                    self.model["keep_probability"]: 0.5,
                }
            )

            if step % 100 == 0:
                self.train_model__report_stats(step)
            else:
                print(".", end="", flush=True)

    def train_model__report_stats(self, step):
        accuracy = self.session.run(
            fetches=self.model["accuracy"],
            feed_dict={
                self.model["x"]: self.preprocess_input(self.input_data.test.images),
                self.model["y"]: self.input_data.test.labels,
                self.model["is_dreaming"]: False,
                self.model["keep_probability"]: 1.,
            }
        )
        print("\n[{}] accuracy: {}".format(step, accuracy))

    def save_trained_values(self, name):
        save_path = self.model["saver"].save(self.session,
                                             '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name))
        print("Model values saved: {}".format(save_path))

    def load_trained_values(self, name):
        checkpoint_path = '{}/{}.ckpt'.format(self.parameters["save_path_prefix"], name)
        self.model["saver"].restore(self.session, checkpoint_path)
        print("Model values restored from checkpoint: {}".format(checkpoint_path))

    def imagine_classes(self):
        pts_num = self.parameters["dream_points_num"]
        # initial_points = np.zeros([pts_num, 2])
        # for i in range():
        #     angle = 2. * math.pi * i / pts_num + 0.1 * 2. * math.pi
        #     initial_points[i, :] += 15. * np.array([math.sin(angle), math.cos(angle)])
        #     initial_points[i, 1] *= 1.3
        #     initial_points[i, :] += [42, 42]
        # initial_points = np.tile(initial_points, [10, 1, 1])
        # initial_points = initial_points.reshape([10, self.parameters["dream_points_num"], 1, 1, 2, 1])

        class_vectors = np.eye(self.parameters["class_num"])
        # print(self.model.keys())
        # print(self.model)
        # self.session.run(
        #     tf.assign(
        #         self.model["dream_points"],
        #         initial_points
        #     )
        # )
        blank_x = np.zeros([1, 28, 28, 1])

        for step in range(self.parameters["dreaming_steps"] + 1):
            if step % 50 == 0:
                self.imagine_classes__report_stats(step, blank_x, class_vectors)
            else:
                print(".", end="", flush=True)

            opt_target = self.model["dreaming_step_2"]

            _ = self.session.run(
                fetches=[
                    # self.model["dream_points"],
                    opt_target,
                ],
                feed_dict={
                    self.model["x"]: blank_x,
                    self.model["y"]: class_vectors,
                    self.model["is_dreaming"]: True,
                    self.model["keep_probability"]: 1.0,
                }
            )

    def imagine_classes__report_stats(self, step, blank_x, class_vectors):

        results, softmax_v = self.session.run(
            fetches=[
                self.model["dream"],
                self.model["result"],
                # self.model["dream_points"]
            ],
            feed_dict={
                self.model["x"]: blank_x,
                self.model["y"]: class_vectors,
                self.model["is_dreaming"]: True,
                self.model["keep_probability"]: 1.0,
            }
        )

        class_num = self.parameters["class_num"]
        probs = {i: softmax_v[i, i] for i in range(class_num)}
        print("\n", step, probs)

        results = np.reshape(results, [-1, self.parameters["input_size"], self.parameters["input_size"]])
        results = np.clip(255 * results, 0, 255).astype(np.uint8)
        left = np.concatenate(results[0:class_num // 2], 0)
        right = np.concatenate(results[class_num // 2:class_num], 0)
        result = np.concatenate([left, right], 1)

        im = Image.fromarray(result)
        im.save("outputs/{}.png".format(step))


def main(argv):
    with tf.Session() as session:
        # sess = tf_debug.LocalCLIDebugWrapperSession(session)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess = session
        mnist = mnist_input.read_data_sets("./data/", one_hot=True)
        trainer = Trainer(sess, mnist)

        should_train = False
        trainer.create_model(should_train)
        if should_train:
            trainer.train_model()
            trainer.save_trained_values("checkpoint1")
        else:
            trainer.load_trained_values("checkpoint1")
            trainer.imagine_classes()


if __name__ == "__main__":
    main(sys.argv[1:])
