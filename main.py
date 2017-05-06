#!/usr/bin/env python
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data as mnist_input


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
        return signal, locals()

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
        return signal, locals()


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
            "dream_points_num": 11,
        }
        if parameters is not None:
            self.parameters.update(parameters)

    def create_dream_from_points(self, dream_points):
        intermediate_points_num = 8


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
        dream_points = tf.Variable(
            tf.random_uniform([dream_count, self.parameters["dream_points_num"], 2], 0., 27.),
            name="dream_points"
        )
        # dream = tf.Variable(tf.zeros([dream_count, size_0, size_0, 1]), name="dream")
        dream, dr1 = self.create_dream_from_points(dream_points)
        signal = tf.cond(is_dreaming, lambda: dream, lambda: x)

        # Layers
        signal, cn1 = self.layers.conv2d_with_bn(signal, 5, 1, 48, "cn1")
        signal = tf.nn.relu(signal)
        signal, mp1 = self.layers.max_pool(signal, 2, "mp1")
        signal, cn2 = self.layers.conv2d_with_bn(signal, 5, 48, 64, "cn2")
        signal = tf.nn.relu(signal)
        signal, mp2 = self.layers.max_pool(signal, 2, "mp2")
        signal = tf.reshape(signal, [-1, 7 * 7 * 64])
        signal, fc1 = self.layers.fully_connected(signal, 7 * 7 * 64, 1024, "fc1")
        signal = tf.nn.relu(signal)
        signal = tf.nn.dropout(signal, keep_probability)
        signal, fc2 = self.layers.fully_connected(signal, 1024, 10, "fc2")
        result = tf.nn.softmax(signal)

        # Measures
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(result), reduction_indices=[1]))
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=signal, name="error"))
        is_correct = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1), name="is_correct")
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")
        contrast = tf.reduce_mean(-tf.multiply(dream, tf.subtract(dream, 1.)))

        # Training operations
        training_step = tf.train.AdamOptimizer(1e-3).minimize(error)
        # training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        dreaming_optimizer = tf.train.AdamOptimizer(1e-2)
        dreaming_step_1 = dreaming_optimizer.minimize(error + contrast, var_list=[dream])
        dreaming_step_2 = dreaming_optimizer.minimize(error, var_list=[dream])

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
        initial_img = Image.open("v_initial.png")
        initial_arr = np.asarray(initial_img, dtype=np.uint8)
        initial_arr = (initial_arr / 255.)
        initial_arr = np.reshape(initial_arr, [28, 28, 1])
        initial_arr = np.broadcast_to(initial_arr, [10, 28, 28, 1])
        # initial_arr = [initial_arr for i in range(10)]
        # initial_arr = 0.5 * np.ones([10, self.parameters["input_size"], self.parameters["input_size"], 1])

        class_vectors = np.eye(self.parameters["class_num"])
        self.session.run(
            tf.assign(
                self.model["dream"],
                initial_arr
            )
        )
        blank_x = np.zeros([1, 28, 28, 1])
        kernel = gkern(3)
        kernel = np.reshape(kernel, [3, 3, 1, 1])
        bkernel = gkern(21, 5.)
        bkernel = np.reshape(bkernel, [21, 21, 1, 1])
        dream = self.model["dream"]
        blur = tf.nn.conv2d(dream, kernel, [1, 1, 1, 1], "SAME")
        treshold = tf.cast(dream > 0.3, tf.float32) * 0.7
        b_threshold = tf.nn.conv2d(treshold, bkernel, [1, 1, 1, 1], "SAME")

        for step in range(self.parameters["dreaming_steps"] + 1):
            # self.session.run([opt_step, ],
            #          feed_dict={x: [blank_image], y_: [class_one_hot], keep_prob: 1.0, is_dreaming: True})

            opt_target = self.model["dreaming_step_1"] if step < 300 else self.model["dreaming_step_2"]
            self.session.run(
                fetches=[
                    opt_target,
                    tf.assign(
                        dream,
                        tf.clip_by_value((0.99 * dream + 0.01 * blur), 0., 1.)
                    )
                ],
                feed_dict={
                    self.model["x"]: blank_x,
                    self.model["y"]: class_vectors,
                    self.model["is_dreaming"]: True,
                    self.model["keep_probability"]: 1.0,
                }
            )

            if step % 200 == 199:
                self.session.run(tf.assign(dream, b_threshold))
                print("Applied threshold!!")

            if step % 50 == 0:
                self.imagine_classes__report_stats(step, blank_x, class_vectors)
            else:
                print(".", end="", flush=True)

    def imagine_classes__report_stats(self, step, blank_x, class_vectors):
        results, softmax_v = self.session.run(
            fetches=[self.model["dream"], self.model["result"]],
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
        print(left.shape, right.shape)
        result = np.concatenate([left, right], 1)
        im = Image.fromarray(result)
        probs = {i: softmax_v[i, i] for i in range(10)}
        print("\n", step, probs)
        im.save("outputs/{}.png".format(step))


def print_usage_and_exit():
    print("Usage: \n"
          "training: solution.py -t\n"
          "visualization: solution.py -v\n")
    sys.exit()


def run_training(trainer):
    print("Running in training mode")
    trainer.run()


def run_visualization(trainer):
    print("Running visualization mode")
    print("Not implemented")


def main(argv):
    print("Modified version of script, again! ")
    with tf.Session() as session:
        mnist = mnist_input.read_data_sets("./data/", one_hot=True)
        trainer = Trainer(session, mnist)
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
