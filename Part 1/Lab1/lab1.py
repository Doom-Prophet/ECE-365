import numpy as np
# import scipy as sp
# import sklearn
# import matplotlib.pyplot as plt


class Lab1(object):
    def solver(self, A, b):
        return np.dot(np.linalg.inv(A),b)

    def fitting(self, x, y):
        xmatr = np.column_stack((x, np.ones(x.shape)))
        coeff = np.dot(np.linalg.pinv(xmatr),y)
        return coeff

    def naive5(self, X, A, Y):
        # Calculate the matrix with $(i,j$)-th entry as  $\mathbf{x}_i^\top A \mathbf{y}_j$ by looping over the rows of $X,Y$.
        qf = np.zeros((X.shape[0],Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                qf[i,j] = np.dot(np.dot(X[i],A), Y[j])
        return qf

    def matrix5(self, X, A, Y):
        # Repeat part (a), but using only matrix operations (no loops!).
        return np.dot(np.dot(X,A), Y.T)

    def naive6(self, X, A):
        # Calculate a vector with $i$-th component $\mathbf{x}_i^\top A \mathbf{x}_i$ by looping over the rows of $X$.
        qf = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            qf[i] = np.dot(np.dot(X[i], A),X[i])
        return qf

    def matrix6(self, X, A):
        # Repeat part (a) using matrix operations (no loops!).
        return np.sum(np.dot(X,A)*X, axis=1)
