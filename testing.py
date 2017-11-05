import numpy as np
from cnn_model import *

layer = np.random.random_integers(-10, 10, size=(1, 100, 100))
filter_size = [2, 3]
stride = [2, 3]
# first dim is the current layer size, second dim is the next layer size, and third dim is the total filter size
filters = [np.random.random_integers(-5, 5, size=(3, 2, 6)), np.random.random_integers(-5, 5, size=(2, 5, 6))]

c1 = convolve(layer, filters[0], filter_size, stride=stride)
m1 = max_pool(c1, [2, 2])
r1 = relu_layer(m1)

c2 = convolve(r1, filters[1], filter_size, stride=stride)
m2 = max_pool(c2, [2, 2])
r2 = relu_layer(m2)

dm2 = back_relu(m2, 1)
dc2 = back_pool(c2, m2, dm2, [2, 2])

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
