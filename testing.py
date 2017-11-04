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