import numpy as np
from cnn_model import *

layer = np.random.random_integers(-10, 10, size=(3, 10, 10))
filter_size = [2, 3]
stride = [2, 3]
layer = np.pad(layer,
               ((0, 0), (int(filter_size[0] / 2), int(filter_size[0] / 2)),
                (int(filter_size[1] / 2), int(filter_size[1] / 2))),
               mode="constant")
# first dim is the current layer size, second dim is the next layer size, and third dim is the total filter size
filters = np.random.random_integers(-5, 5, size=(3, 2, 6))
print(layer)
print(filters)
c1 = convolve(layer, filters, filter_size, stride=stride)
print(c1)
m1 = max_pool(c1, [2, 2])
print(m1)
r1 = relu_layer(m1)
print(r1)

'''
A = np.array([[[1, 1, 2],
               [1, 2, 3],
               [2, 3, 1]],

              [[1, 2, 2],
               [1, 2, 3],
               [2, 3, 1]]
             ])
B = np.array([[[1, 1, 1],
               [1, 1, 1]],

              [[1, 1, 1],
               [0, 0, 0]]
              ])
print(np.array([np.dot(B[i], A[i]) for i in range(len(A))]))
print(np.sum(np.array([np.dot(B[i], A[i]) for i in range(len(A))]), axis=2))
'''
