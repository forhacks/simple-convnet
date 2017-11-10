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
    layer = pad_layer(layer, [(int(filter_size[0] / 2), int(filter_size[0] / 2)),
                              (int(filter_size[1] / 2), int(filter_size[1] / 2))])
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
    layer = pad_layer(layer, [(0, -len(layer[0]) % size[0]),
                              (0, -len(layer[0][0]) % size[1])])
    new_layer_size = (len(layer),
                      int((len(layer[0]) - size[0]) / stride[0] + 1),
                      int((len(layer[0][0]) - size[1]) / stride[1] + 1))
    layer = im2col(layer, size, stride)
    layer = np.amax(np.reshape(layer, (new_layer_size[0], size[0] * size[1], -1)), axis=1)
    layer = np.resize(layer, new_layer_size)
    return layer


def relu_layer(layer):
    gradients = np.maximum(layer, layer*0.01)
    return gradients


def back_relu(layer, deriv):
    layer = np.where(layer < 0, 0.01, 1)
    return np.multiply(layer, deriv)


# TODO get the backpropagation for pooling to work
def back_pool(layer, next_layer, deriv, size, stride=None):
    if stride is None:
        stride = size
    repeated = np.repeat(np.repeat(next_layer, size[1], axis=2), size[0], axis=1)
    repeated = repeated[::, :len(layer[0]):, :len(layer[0][0]):]
    layer = layer - repeated
    layer = np.where(layer < 0, 0, 1)
    repeated = np.repeat(np.repeat(deriv, size[1], axis=2), size[0], axis=1)
    repeated = repeated[::, :len(layer[0]):, :len(layer[0][0]):]
    prev_layer = np.multiply(layer, repeated)
    return prev_layer


def back_conv(layer, deriv, size, stride=[1, 1]):
    derivW = np.resize(deriv, (len(deriv), len(deriv[0]) * len(deriv[0][0])))
    print(deriv.shape)
    print(derivW.shape)
    dw = np.zeros((len(layer), len(deriv), size[0] * size[1]))
    for i in range(len(layer)):
        for j in range(len(deriv)):
                dw[i][j] += convolve(np.array([layer[i]]),
                                     np.array([[derivW[j]]]),
                                     (deriv.shape[1], deriv.shape[2]),
                                     stride=stride)
    print(dw)


def pad_layer(layer, size):
    return np.pad(layer, ((0, 0), (size[0]), (size[1])), mode="constant")