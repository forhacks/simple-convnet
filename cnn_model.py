import numpy as np


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
    layer = np.pad(layer,
                   ((int(filter_size[0]/2), int(filter_size[0]/2)), (int(filter_size[1]/2), int(filter_size[1]/2))),
                   mode="constant")
    layer = im2col(layer, filter_size, stride)
    return np.sum(np.array([np.dot(filters[i], layer[i]) for i in range(len(layer))]), axis=2)
