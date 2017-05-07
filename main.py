#!/usr/bin/env python
import sys

import math
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data as mnist_input
from tensorflow.python import debug as tf_debug


def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel / np.sum(kernel)


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
            "dream_points_num": 7,
        }
        if parameters is not None:
            self.parameters.update(parameters)

    def create_model(self):
        print("Creating the model")
        size_0 = self.parameters["input_size"]
        dream_count = self.parameters["class_num"]

        # Model switches
        is_dreaming = tf.placeholder(tf.bool, name="is_dreaming")
        keep_probability = tf.placeholder(tf.float32, name="keep_probability")

        # Network placeholders and gates
        x = tf.placeholder(tf.float32, [None, size_0, size_0, 1], name="x")
        y = tf.placeholder(tf.float32, [None, 10], name="y")

        dream_points_num = self.parameters["dream_points_num"]
        # line_segments_num = dream_points_num // 2
        line_segments_num = dream_points_num - 1  #- 5
        big_dream_size = 28 * 3
        dream_points = tf.Variable(
            tf.random_uniform([dream_count, dream_points_num, 1, 1, 2, 1], 0., big_dream_size),
            name="dream_points"
        )

        coord1 = np.broadcast_to(np.arange(big_dream_size), [big_dream_size, big_dream_size])
        coord2 = coord1.T

        coords = np.stack([coord1, coord2], 2)  # shape [bds, bds, 2]
        coords = np.reshape(coords, [1, 1, big_dream_size, big_dream_size, 2, 1])

        # p1 = dream_points[:, 0::2, :, :]
        # p1 = p1[:, :-1, :, :]
        # p2 = dream_points[:, 1::2, :, :]

        # p1 = dream_points[:, :-6, :, :]
        # p2 = dream_points[:, 1:-5, :, :]

        p1 = dream_points[:, :-1, :, :]
        p2 = dream_points[:, 1:, :, :]

        _v = coords - p1
        _s = p1 - p2
        _s = tf.tile(_s, [1, 1, big_dream_size, big_dream_size, 1, 1])
        g = tf.matmul(_v, _s, True)
        d = tf.matmul(_s, _s, True)
        rate = g / (d + 1e-6)
        proj = rate * _s
        proj = proj + p1

        z = p1 + p2 - 2 * proj
        d2 = tf.matmul(z, z, True)
        projects_on_segment = tf.sigmoid((tf.sqrt(d + 1e-6) - tf.sqrt(d2 + 1e-6)) * 0.25)

        # projects_on_segment = tf.sigmoid(d - d2)
        projects_on_segment = tf.reshape(projects_on_segment,
                                         [dream_count, line_segments_num, big_dream_size, big_dream_size, 1])

        vecs = coords - proj
        dots = tf.matmul(vecs, vecs, True)
        dist = tf.sqrt(dots + 1e-6)
        is_close_to_segment = tf.reduce_max(tf.sigmoid((3. - dist) * 0.5), [4])

        big_dream = projects_on_segment * is_close_to_segment
        big_dream = tf.reduce_mean(big_dream, 1)
        # big_dream = tf.reduce_max(tf.sigmoid(proj - p1) * tf.sigmoid(p2 - proj), [1, 4])
        print("big_dream.shape", big_dream.shape)
        dream = tf.image.resize_bilinear(big_dream, [28, 28], name="dream")

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
        error = cross_entropy
        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(result), reduction_indices=[1]))
        # error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=signal, name="error"))
        is_correct = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1), name="is_correct")
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")
        contrast = tf.reduce_mean(-tf.multiply(dream, tf.subtract(dream, 1.)))

        # Training operations
        training_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, var_list=v1 + v2 + v3 + v4)
        # training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        dreaming_optimizer = tf.train.AdamOptimizer(0.05)
        # dreaming_optimizer = tf.train.GradientDescentOptimizer(0.0000000000000001)
        black_ratio = 1. - tf.reduce_mean(dream)
        optimize_white_balance = dreaming_optimizer.minimize(cross_entropy + black_ratio, var_list=[dream_points])
        dreaming_step_1 = dreaming_optimizer.minimize(cross_entropy + contrast, var_list=[dream_points])
        dreaming_step_2 = dreaming_optimizer.minimize(cross_entropy, var_list=[dream_points])

        saver = tf.train.Saver()
        self.model = locals()

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
        # initial_img = Image.open("initial_test3.png")
        # initial_img = Image.open("v_initial.png")
        # initial_arr = np.asarray(initial_img, dtype=np.uint8)
        # initial_arr = (initial_arr / 255.)
        # initial_arr = np.reshape(initial_arr, [28, 28, 1])
        # initial_arr = np.broadcast_to(initial_arr, [10, 28, 28, 1])
        # initial_arr = [initial_arr for i in range(10)]
        # initial_arr = 0.5 * np.ones([10, self.parameters["input_size"], self.parameters["input_size"], 1])

        # 10, 7, 1, 1, 2, 1
        initial_points = np.zeros([self.parameters["dream_points_num"], 2])
        # for i in range(self.parameters["dream_points_num"]):
        #     angle = 2. * math.pi * i / 7 + 0.1 * 2. * math.pi
        #     initial_points[i, :] += 15. * np.array([math.sin(angle), math.cos(angle)])
        #     initial_points[i, 1] *= 1.3
        #     initial_points[i, :] += [42, 42]
        initial_points[0, :] = [42, 10]
        initial_points[1, :] = [62, 30]
        initial_points[2, :] = [22, 54]
        initial_points[3, :] = [42, 74]
        initial_points[4, :] = [62, 54]
        initial_points[5, :] = [22, 30]
        initial_points[6, :] = [42, 10]
        # initial_points[7,:] = [79, 39]
        initial_points = np.tile(initial_points, [10, 1, 1])
        initial_points = initial_points.reshape([10, self.parameters["dream_points_num"], 1, 1, 2, 1])
        # initial_points = tf.random_normal([10, 7, 1, 1, 2, 1], 42., 30.)
        class_vectors = np.eye(self.parameters["class_num"])
        self.session.run(
            tf.assign(
                self.model["dream_points"],
                initial_points
            )
        )
        blank_x = np.zeros([1, 28, 28, 1])
        # kernel = gkern(3)
        # kernel = np.reshape(kernel, [3, 3, 1, 1])
        # bkernel = gkern(21, 5.)
        # bkernel = np.reshape(bkernel, [21, 21, 1, 1])
        # dream = self.model["dream"]
        # blur = tf.nn.conv2d(dream, kernel, [1, 1, 1, 1], "SAME")
        # treshold = tf.cast(dream > 0.3, tf.float32) * 0.7
        # b_threshold = tf.nn.conv2d(treshold, bkernel, [1, 1, 1, 1], "SAME")

        for step in range(self.parameters["dreaming_steps"] + 1):
            # for step in range(1):
            # self.session.run([opt_step, ],
            #          feed_dict={x: [blank_image], y_: [class_one_hot], keep_prob: 1.0, is_dreaming: True})

            # opt_target = self.model["dreaming_step_1"] if step < 300 else self.model["dreaming_step_2"]
            if step % 50 == 0:
                self.imagine_classes__report_stats(step, blank_x, class_vectors)
            else:
                print(".", end="", flush=True)

            opt_target = self.model["dreaming_step_2"]
            # opt_target = self.model["optimize_white_balance"]

            pts, _ = self.session.run(
                fetches=[
                    self.model["dream_points"],
                    opt_target,
                    # tf.assign(
                    #     dream,
                    #     tf.clip_by_value((0.99 * dream + 0.01 * blur), 0., 1.)
                    # )
                ],
                feed_dict={
                    self.model["x"]: blank_x,
                    self.model["y"]: class_vectors,
                    self.model["is_dreaming"]: True,
                    self.model["keep_probability"]: 1.0,
                }
            )

            # print(pts)

            # if step % 200 == 199:
            #     self.session.run(tf.assign(dream, b_threshold))
            #     print("Applied threshold!!")

    def imagine_classes__report_stats(self, step, blank_x, class_vectors):
        # x = self.model["dream_points"]
        # y = self.model["error"]

        # print(x.shape)
        # print(y.shape)
        # print(tf.test.compute_gradient(
        #     x, (10, 7, 1, 1, 2, 1), y, (), delta=0.001, extra_feed_dict={
        #         self.model["is_dreaming"]: True,
        #         self.model["keep_probability"]: 1.0,
        #         self.model["x"]: blank_x,
        #         self.model["y"]: class_vectors,
        #     }
        # ))
        # print(tf.test.compute_gradient_error(
        #     x, (10, 7, 1, 1, 2, 1), y, (), delta=0.001, extra_feed_dict={
        #         self.model["is_dreaming"]: True,
        #         self.model["keep_probability"]: 1.0,
        #         self.model["x"]: blank_x,
        #         self.model["y"]: class_vectors,
        #     }
        # ))

        results, softmax_v, pts = self.session.run(
            fetches=[self.model["dream"], self.model["result"], self.model["dream_points"]],
            feed_dict={
                self.model["x"]: blank_x,
                self.model["y"]: class_vectors,
                self.model["is_dreaming"]: True,
                self.model["keep_probability"]: 1.0,
            }
        )

        results = np.reshape(results, [-1, self.parameters["input_size"], self.parameters["input_size"]])
        results = np.clip(255 * results, 0, 255).astype(np.uint8)
        left = np.concatenate(results[0:5], 0)
        right = np.concatenate(results[5:10], 0)
        result = np.concatenate([left, right], 1)
        im = Image.fromarray(result)
        probs = {i: softmax_v[i, i] for i in range(10)}
        print("\n", step, probs)
        im.save("outputs/{}.png".format(step))


def main(argv):
    print("Modified version of script, again! ")
    with tf.Session() as session:
        # sess = tf_debug.LocalCLIDebugWrapperSession(session)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess = session
        mnist = mnist_input.read_data_sets("./data/", one_hot=True)
        trainer = Trainer(sess, mnist)
        trainer.create_model()
        should_train = False
        if should_train:
            trainer.train_model()
            trainer.save_trained_values("checkpoint1")
        else:
            trainer.load_trained_values("checkpoint1")
            trainer.imagine_classes()


if __name__ == "__main__":
    main(sys.argv[1:])
