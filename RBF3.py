from scipy import *
from scipy.linalg import norm, pinv

from matplotlib import pyplot as plt


class RBF:

    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(0.01, 1.5, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)

         # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y


def main():
    n = 100
    x = mgrid[0:10:complex(0, n)].reshape(n, 1)
    y = 0.5 + 0.3 * sin(2 * x)
    # rbf regression
    rbf = RBF(1, 100, 1)
    rbf.train(x,y)
    z = rbf.test(x)

    # plot original data
    plt.figure()
    plt.plot(x,y,'k-')
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x,z,'r-',linewidth=2)

    # plot rbfs
    # plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
    '''for c in rbf.centers:
        # RF prediction lines
        cx = arange(c - 0.14, c + 0.14, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)'''
    # plt.xlim(-1.2, 1.2)
    plt.show()
