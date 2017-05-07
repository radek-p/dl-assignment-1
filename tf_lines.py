import tensorflow as tf
import numpy as np
from PIL import Image

dream_points_num = 3
dream_count = 2

line_segments_num = dream_points_num - 1
dream_points = tf.Variable(
    tf.random_uniform([dream_count, dream_points_num, 1, 1, 2, 1], 0., 27.),
    name="dream_points"
)

big_dream_size = 28 * 3
coord1 = np.broadcast_to(np.arange(big_dream_size), [big_dream_size, big_dream_size])
coord2 = coord1.T

coords = np.stack([coord1, coord2], 2)  # shape [bds, bds, 2]
coords = np.reshape(coords, [1, 1, big_dream_size, big_dream_size, 2, 1])

p1 = dream_points[:, :-1, :, :]
p2 = dream_points[:, 1:, :, :]
_v = coords - p1
_s = p1 - p2
_s = tf.tile(_s, [1, 1, big_dream_size, big_dream_size, 1, 1])
g = tf.matmul(_v, _s, True)
d = tf.matmul(_s, _s, True)
rate = g / d
proj = rate * _s
proj = proj + p1

# big_dream = tf.logical_or(
#     tf.logical_and(
#         tf.less(p1[:, :, :, :, 0], proj[:, :, :, :, 0]),
#         tf.less(proj[:, :, :, :, 0], p2[:, :, :, :, 0])
#     ),
#     tf.logical_and(
#         tf.less(p1[:, :, :, :, 1], proj[:, :, :, :, 1]),
#         tf.less(proj[:, :, :, :, 1], p2[:, :, :, :, 1])
#     )
# )
# big_dream = tf.cast(big_dream, tf.float32)

z = p1 + p2 - 2 * proj
d2 = tf.matmul(z, z, True)

# cond1 = tf.sigmoid(proj - p1 - 3.)
# cond2 = tf.sigmoid(p2 - proj - 3.)
# print("cond1", cond1.shape)
# dcond1x = cond1[:, 0, :, :, 0, :]
# dcond2x = cond2[:, 0, :, :, 0, :]
# dcond1y = cond1[:, 0, :, :, 1, :]
# dcond2y = cond2[:, 0, :, :, 1, :]
# projects_on_segment = tf.reduce_max(cond1 * cond2, [4])
# projects_on_segment = tf.sigmoid(tf.sqrt(d) - tf.sqrt(d2))
projects_on_segment = tf.sigmoid((tf.sqrt(d + 1e-6) - tf.sqrt(d2 + 1e-6)) * 0.125)
print("projects:", projects_on_segment.shape)
projects_on_segment = tf.reshape(projects_on_segment, [dream_count, line_segments_num, big_dream_size, big_dream_size, 1])

vecs = coords - proj
print("vecs: ", vecs.shape)
dots = tf.matmul(vecs, vecs, True)
print("dots: ", dots.shape)
dist = tf.sqrt(dots + 1e-6)
is_close_to_segment = tf.reduce_max(tf.sigmoid((3. - dist) * 0.5), [4])

big_dream = projects_on_segment * is_close_to_segment
big_dream = tf.reduce_max(big_dream, 1)

initial_points = np.zeros([dream_count, dream_points_num, 2])

# for i in range(dream_points_num):
#     # if i % 2 == 0:
#     #     initial_points[i, 0] = 30.
#     # else:
#     #     initial_points[i, 0] = 54.
#     initial_points[i, 0] = 9. + i * 11.
#     initial_points[i, 1] = 9. + i * 11.

initial_points[0, 0, :] = [10, 10]
initial_points[0, 1, :] = [10, 60]
initial_points[0, 2, :] = [80, 50]

initial_points[1, 0, :] = [10, 10]
initial_points[1, 1, :] = [11, 60]
initial_points[1, 2, :] = [80, 40]

# initial_points[2, 0, :] = [10, 10]
# initial_points[2, 1, :] = [15, 60]

# initial_points[3, 0, :] = [10, 10]
# initial_points[3, 1, :] = [20, 60]
#
# initial_points[4, 0, :] = [10, 10]
# initial_points[4, 1, :] = [30, 60]
#
# initial_points[5, 0, :] = [10, 10]
# initial_points[5, 1, :] = [40, 60]

# initial_points[2, :] = [40, 4]
# initial_points[3, :] = [20, 6]
# initial_points[4, :] = [23, 8]
# initial_points[5, :] = [24, 8]
# initial_points[6, :] = [25, 8]

# initial_points = np.tile(initial_points, [dream_count, 1, 1])
initial_points = initial_points.reshape([dream_count, dream_points_num, 1, 1, 2, 1])
print("init_sh", initial_points.shape)
print(initial_points)

class_vectors = np.eye(dream_count)

session = tf.InteractiveSession()

session.run(
    tf.assign(
        dream_points,
        initial_points
    )
)

whole, p1, p2 = session.run(
    fetches=[big_dream, projects_on_segment, is_close_to_segment],
    feed_dict={}
)


def save(results, name):
    results = np.reshape(results, [-1, big_dream_size, big_dream_size])
    results = np.clip(255 * results, 0, 255).astype(np.uint8)
    left = np.concatenate(results[0:dream_count // 2], 0)
    right = np.concatenate(results[dream_count // 2:dream_count], 0)
    print(left.shape, right.shape)
    result = np.concatenate([left, right], 1)
    im = Image.fromarray(result)
    im.save(name)


save(whole, "lines.png")
# save(p1, "p1.png")
# save(p2, "p2.png")
# save(d1x, "d1x.png")
# save(d2x, "d2x.png")
# save(d1y, "d1y.png")
# save(d2y, "d2y.png")
