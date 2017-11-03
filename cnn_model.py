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


# def convolve():
#
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
