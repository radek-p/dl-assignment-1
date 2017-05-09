#!/usr/bin/env python
import argparse
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data as mnist_input


class LayerBuilder(object):
    def __init__(self, parameters=None):
        self.parameters = {
            "epsilon": 1e-3,
        }
        if parameters is not None:
            self.parameters.update(parameters)

    def conv2d_with_bn(self, signal, filter_size, in_channels, out_channels, is_training, basename="conv_layer"):
        # create trainable variables for the layer
        W = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1),
            name="{}_W".format(basename)
        )
        gamma = tf.Variable(tf.constant(1.0, shape=[out_channels]), name="{}_gamma".format(basename))
        beta = tf.Variable(tf.constant(0.0, shape=[out_channels]), name="{}_beta".format(basename))

        # apply convolution filters
        signal = tf.nn.conv2d(signal, W, strides=[1, 1, 1, 1], padding="SAME")

        moving_average_obj = tf.train.ExponentialMovingAverage(decay=0.9999)

        # normalize
        batch_mean = tf.reduce_mean(signal, axis=[0, 1, 2])
        trained_mean = tf.Variable(tf.constant(0.2, shape=[out_channels]), False,
                                   name="{}_trained_mean".format(basename))
        # trained_mean = moving_average_obj.average(batch_mean)
        mean_update = tf.assign(trained_mean, trained_mean * 0.9999 + batch_mean * 0.0001)
        signal -= tf.cond(is_training, lambda: batch_mean, lambda: trained_mean)

        batch_variance = tf.reduce_mean(tf.square(signal), axis=[0, 1, 2])
        trained_variance = tf.Variable(tf.constant(0.4, shape=[out_channels]), False,
                                       name="{}_trained_variance".format(basename))
        # trained_variance = moving_average_obj.average(batch_variance)
        variance_update = tf.assign(trained_variance, trained_variance * 0.9999 + batch_variance * 0.0001)
        variance = tf.cond(is_training, lambda: batch_variance, lambda: trained_variance)
        signal /= tf.sqrt(variance + self.parameters["epsilon"])

        # update_moving_average_op = moving_average_obj.apply([batch_mean, batch_variance])

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
            "line_segments_num": 31,
            "batch_size": 100,
        }
        if parameters is not None:
            self.parameters.update(parameters)

    @staticmethod
    def prepare_initial_line_segments():
        # Draw circle
        rot = np.array([[0, -1], [1, 0]])
        circle_points = [np.array([0, -6]), np.array([4.5, -4.5])]
        for _ in range(7):
            circle_points.append(np.matmul(rot, circle_points[-2]))
        circle1 = np.array(list(zip(circle_points[:-1], circle_points[1:])), dtype=np.float32)
        circle2 = circle1.copy()
        circle1[:, :, 1] -= 6
        circle2[:, :, 1] += 6

        # draw rectangular frame
        horizontal_lines = [[(x, y), (0, y)] for y in [-12, 0, 12] for x in [-6, 6]]
        vertical_lines = [[(x, y), (x, y + 6)] for y in [-12, -6, 0, 6] for x in [-6, 6]]
        frame = np.array(horizontal_lines + vertical_lines, dtype=np.float32)

        # draw diagonal
        diagonal = np.array([[(6, -12), (0, 12)]], dtype=np.float32)

        line_segments = np.concatenate((circle1, circle2, frame, diagonal), 0)
        line_segments *= 0.66  # .74, .70
        return line_segments

    def create_dream(self, create_variables_only):
        class_num = self.parameters["class_num"]
        line_segments_num = self.parameters["line_segments_num"]
        canvas_size = self.parameters["input_size"]
        epsilon = 1e-6

        scales = tf.Variable(tf.ones([class_num, line_segments_num, 1, 1, 1, 1]))
        opacities = tf.Variable(tf.ones([class_num, line_segments_num, 1, 1, 1]))

        trainable_variables = [scales, opacities]

        if create_variables_only:
            # This is a hack to make network training faster.
            # In a training mode we don't have to create graph nodes that are only used in the dreaming phase.
            # However, variables related to dreaming still need to be somehow connected to the main graph,
            # or they would not be saved, this would cause problems in second phase.
            canvas = tf.ones([class_num, canvas_size, canvas_size, 1]) \
                     * scales[0, 0, 0, 0, 0] * opacities[0, 0, 0, 0, 0]
            return canvas, locals(), trainable_variables

        coord1 = np.broadcast_to(np.arange(canvas_size, dtype=np.float32), [canvas_size, canvas_size])
        coord2 = coord1.T

        coordinates = np.stack([coord1, coord2], 2)
        coordinates = np.reshape(coordinates, [1, 1, canvas_size, canvas_size, 2, 1])

        line_segments = self.prepare_initial_line_segments()
        line_segments = np.broadcast_to(line_segments, [class_num, line_segments_num, 2, 2])
        line_segments = line_segments * tf.reshape(scales, [class_num, line_segments_num, 1, 1])
        line_segments += np.array([13, 13])
        line_segments = tf.reshape(line_segments, [class_num, line_segments_num, 1, 1, 2, 2, 1])
        p1, p2 = tf.unstack(line_segments, axis=4)

        v = coordinates - p1
        s = p1 - p2
        s = tf.tile(s, [1, 1, canvas_size, canvas_size, 1, 1])
        s_norm2 = tf.matmul(s, s, True)
        projection = tf.matmul(v, s, True) / (s_norm2 + epsilon) * s
        projection = projection + p1

        z = p1 + p2 - 2 * projection
        z_norm2 = tf.matmul(z, z, True)

        projects_on_segment = tf.sigmoid((tf.sqrt(s_norm2 + epsilon) - tf.sqrt(z_norm2 + epsilon)) * 2.)
        projects_on_segment = tf.reshape(projects_on_segment,
                                         [class_num, line_segments_num, canvas_size, canvas_size, 1])

        l = coordinates - projection
        l_norm = tf.sqrt(tf.matmul(l, l, True) + epsilon)
        is_close_to_segment = tf.reduce_max(tf.sigmoid((0.5 - l_norm) * 2.), [4])

        canvas = projects_on_segment * is_close_to_segment * opacities
        canvas = tf.clip_by_value(tf.reduce_sum(canvas, 1), 0., 1.)

        return canvas, locals(), trainable_variables

    def create_model(self, trimmed_for_faster_training):
        print("Creating the model")
        input_size = self.parameters["input_size"]

        # Model switches
        is_dreaming = tf.placeholder(tf.bool, name="is_dreaming")
        is_training = tf.placeholder(tf.bool, name="is_training")
        keep_probability = tf.placeholder(tf.float32, name="keep_probability")

        # Network placeholders and gates
        x = tf.placeholder(tf.float32, [None, input_size, input_size, 1], name="x")
        y = tf.placeholder(tf.float32, [None, 10], name="y")

        canvas, init, v0 = self.create_dream(trimmed_for_faster_training)

        signal = tf.cond(is_dreaming, lambda: canvas, lambda: x)

        # Layers
        signal, conv1, v1 = self.layers.conv2d_with_bn(signal, 5, 1, 48, is_training, basename="conv1")
        signal = tf.nn.relu(signal)
        signal, mp1 = self.layers.max_pool(signal, 2, basename="mp1")
        signal, conv2, v2 = self.layers.conv2d_with_bn(signal, 5, 48, 64, is_training, basename="conv2")
        signal = tf.nn.relu(signal)
        signal, mp2 = self.layers.max_pool(signal, 2, basename="mp2")
        signal = tf.reshape(signal, [-1, 7 * 7 * 64])
        signal, fc1, v3 = self.layers.fully_connected(signal, 7 * 7 * 64, 1024, basename="fc1")
        signal = tf.nn.relu(signal)
        signal = tf.nn.dropout(signal, keep_probability, name="dropout")
        signal, fc2, v4 = self.layers.fully_connected(signal, 1024, 10, basename="fc2")
        result = tf.nn.softmax(signal)

        # Measures
        # cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(result, 1e-10, 1.0)))
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=signal, name="error"))
        is_correct = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1), name="is_correct")
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")
        white_ratio = tf.reduce_mean(canvas)

        # Training operations
        training_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, var_list=v1 + v2 + v3 + v4)

        dreaming_optimizer = tf.train.AdamOptimizer(0.05, name="dreaming_optimizer")
        dreaming_step_1 = dreaming_optimizer.minimize(cross_entropy + tf.square(white_ratio - 0.11) * 100, var_list=v0)
        dreaming_step_2 = dreaming_optimizer.minimize(cross_entropy, var_list=v0)
        second_part_of_the_hack = dreaming_optimizer.minimize(cross_entropy, var_list=v1 + v2 + v3 + v4)

        saver = tf.train.Saver()
        self.model = locals()

    def preprocess_input(self, x):
        return np.reshape(x, [-1, self.parameters["input_size"], self.parameters["input_size"], 1])

    def train_model(self):
        print("Training the model")
        self.session.run(tf.global_variables_initializer())

        for step in range(self.parameters["training_steps"] + 1):
            x, y = self.input_data.train.next_batch(self.parameters["batch_size"])
            x = self.preprocess_input(x)

            self.session.run(
                fetches=[
                    self.model["training_step"],
                    self.model["conv1"]["mean_update"],
                    self.model["conv2"]["mean_update"],
                    self.model["conv1"]["variance_update"],
                    self.model["conv2"]["variance_update"],
                ],
                feed_dict={
                    self.model["x"]: x,
                    self.model["y"]: y,
                    self.model["is_dreaming"]: False,
                    self.model["is_training"]: True,
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
                self.model["is_training"]: True,
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
        class_vectors = np.eye(self.parameters["class_num"])
        blank_x = np.zeros([1, 28, 28, 1])

        for step in range(self.parameters["dreaming_steps"] + 1):
            if step % 50 == 0:
                self.imagine_classes__report_stats(step, blank_x, class_vectors)
            else:
                print(".", end="", flush=True)

            opt_target = self.model["dreaming_step_2"]

            self.session.run(
                fetches=[
                    opt_target,
                ],
                feed_dict={
                    self.model["x"]: blank_x,
                    self.model["y"]: class_vectors,
                    self.model["is_dreaming"]: True,
                    self.model["is_training"]: False,
                    self.model["keep_probability"]: 1.0,
                }
            )

    def imagine_classes__report_stats(self, step, blank_x, class_vectors):

        wr, results, softmax_v = self.session.run(
            fetches=[
                self.model["white_ratio"],
                self.model["canvas"],
                self.model["result"],
            ],
            feed_dict={
                self.model["x"]: blank_x,
                self.model["y"]: class_vectors,
                self.model["is_dreaming"]: True,
                self.model["is_training"]: False,
                self.model["keep_probability"]: 1.0,
            }
        )

        class_num = self.parameters["class_num"]
        probs = {i: softmax_v[i, i] for i in range(class_num)}
        print("\n", step, ", wr: ", wr, probs)

        results = np.reshape(results, [-1, self.parameters["input_size"], self.parameters["input_size"]])
        results = np.clip(255 * results, 0, 255).astype(np.uint8)
        left = np.concatenate(results[0:class_num // 2], 0)
        right = np.concatenate(results[class_num // 2:class_num], 0)
        result = np.concatenate([left, right], 1)

        im = Image.fromarray(result)
        im.save("outputs/{}.png".format(step))


def main(argv):
    parser = argparse.ArgumentParser(prog='main.py')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--training', action='store_true', default=False,
                       help="the model is created and trained from scratch")
    group.add_argument('-d', '--dreaming', action='store_true', default=False,
                       help="pre-trained model is used to create images that maximize class probabilities")
    parser.add_argument('checkpoint_dir',
                        help="directory that the checkpoint should be stored in / loaded from (it must exist!)")
    parser.add_argument('checkpoint_name',
                        help="name of a checkpoint (without .ckpt suffix)")
    options = parser.parse_args(argv)

    with tf.Session() as session:
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(session)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess = session
        mnist = mnist_input.read_data_sets("./data/", one_hot=True)
        trainer = Trainer(sess, mnist, parameters={
            "save_path_prefix": options.checkpoint_dir
        })

        trainer.create_model(options.training)
        if options.training:
            trainer.train_model()
            trainer.save_trained_values(options.checkpoint_name)
        else:
            trainer.load_trained_values(options.checkpoint_name)
            trainer.imagine_classes()


if __name__ == "__main__":
    main(sys.argv[1:])
