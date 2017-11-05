import numpy as np
from skimage.util.shape import view_as_blocks
import math


def im2col(A, B, skip):
    # Parameters
    D, M, N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(D)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(A, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    return out


def convolve(layer, filters, filter_size, stride=[1, 1]):
    layer_size = len(layer)
    new_layer_size = (len(filters[0]),
                      int((len(layer[0]) - filter_size[0]) / stride[0] + 1),
                      int((len(layer[0][0]) - filter_size[1]) / stride[1] + 1))
    # print(layer)
    layer = im2col(layer, filter_size, stride)
    # print(layer)
    layer = np.reshape(layer, (layer_size, filter_size[0] * filter_size[1], -1))
    # print(layer)
    # print(filters)
    new_layer = np.sum(np.array([np.dot(filters[i], layer[i]) for i in range(len(layer))]), axis=0)
    # print(new_layer)
    return np.resize(new_layer, new_layer_size)


def max_pool(layer, size, stride=None):
    if stride is None:
        stride = size
    new_layer_size = (len(layer),
                      int((len(layer[0]) - size[0]) / stride[0] + 1),
                      int((len(layer[0][0]) - size[1]) / stride[1] + 1))
    layer = im2col(layer, size, stride)
    layer = np.amax(np.reshape(layer, (new_layer_size[0], size[0] * size[1], -1)), axis=1)
    layer = np.resize(layer, new_layer_size)
    return layer


def relu_layer(layer):
    gradients = np.maximum(layer, 0.01*layer)
    return gradients