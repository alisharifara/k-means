from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators import kmeans

input_1d_x = np.array([130,150,120,190,187,184,110,167,143,193,194,120,122,153,100,200])

def input_fn_1d(input):
    input_t = tf.convert_to_tensor(input, dtype=tf.float32)
    input_t = tf.expand_dims(input_t, 1)
    
    return (input_t, None)
    
plt.scatter(input_1d_x, np.zeros_like(input_1d_x), s=500)
plt.show()


k_means_estimator = kmeans.KMeansClustering(num_clusters = 2)

clusters_1d = k_means_estimator.clusters()
print(clusters_1d)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(input_1d_x, np.zeros_like(input_1d_x), s=300, marker='o')
ax1.scatter(clusters_1d, np.zeros_like(clusters_1d), c='r', s=300, marker='s')

k_means_estimator.get_params()

